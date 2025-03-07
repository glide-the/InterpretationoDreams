import threading
import queue
from dreamsboard.engine.storage.task_step_store.simple_task_step_store import SimpleTaskStepStore
from langchain_community.chat_models import ChatOpenAI
from dreamsboard.dreams.builder_task_step.base import StructuredTaskStepStoryboard
from dreamsboard.engine.utils import concat_dirs
from dreamsboard.engine.storage.task_step_store.types import DEFAULT_PERSIST_FNAME
from dreamsboard.common.try_parse_json_object import try_parse_json_object
 
from dreamsboard.engine.entity.task_step.task_step import RefineResponse
from dreamsboard.dreams.task_step_md.base import TaskStepMD
from dreamsboard.common import _get_assistants_tool

from dreamsboard.engine.storage.task_step_store.types import BaseTaskStepStore
from dreamsboard.engine.task_engine_builder.core import TaskEngineBuilder
import logging

from dreamsboard.utils import get_config_dict, get_log_file, get_timestamp_ms
import logging.config
import os
from dreamsboard.dreams.task_step_to_question_chain.weaviate.prepare_load import get_query_hash
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)

logging_conf = get_config_dict(
    "DEBUG",
    get_log_file(log_path="logs", sub_dir=f"local_{get_timestamp_ms()}"),
    100*1024*1024, 
    100*1024*1024,
)

logging.config.dictConfig(logging_conf)  # type: ignore

llm = ChatOpenAI(
    openai_api_base=os.environ.get("ZHIPUAI_API_BASE"),
    model=os.environ.get("ZHIPUAI_API_MODEL"),
    openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
    verbose=True,
    temperature=0.1,
    top_p=0.9,
) 

guiji_llm = ChatOpenAI(
    openai_api_base=os.environ.get("ZHIPUAI_API_BASE"),
    model=os.environ.get("ZHIPUAI_API_MODEL"),
    openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
    verbose=True,
    temperature=0.1,
    top_p=0.9,
) 
if 'glm' in os.environ.get("ZHIPUAI_API_MODEL"):

    tools= [ { "type": "web_search",   "web_search": {"enable": False ,"search_result": False   }}]
else:
    tools = []
llm_with_tools = llm.bind(   tools=[_get_assistants_tool(tool) for tool in tools] )
kor_dreams_task_step_llm_with_tools = guiji_llm.bind(   tools=[_get_assistants_tool(tool) for tool in tools] )


from tests.test_builder_task_step.prompts import (
    AEMO_REPRESENTATION_PROMPT_TEMPLATE as AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST,
    STORY_BOARD_SCENE_TEMPLATE as STORY_BOARD_SCENE_TEMPLATE_TEST,
    STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE as STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST,
    EDREAMS_EVOLUTIONARY_TEMPLATE as EDREAMS_EVOLUTIONARY_TEMPLATE_TEST,
    EDREAMS_PERSONALITY_TEMPLATE as EDREAMS_PERSONALITY_TEMPLATE_TEST,
    DREAMS_GEN_TEMPLATE as DREAMS_GEN_TEMPLATE_TEST,
) 
os.environ["AEMO_REPRESENTATION_PROMPT_TEMPLATE"] = AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
os.environ["STORY_BOARD_SCENE_TEMPLATE"] = STORY_BOARD_SCENE_TEMPLATE_TEST
os.environ["STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE"] = STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST
os.environ["EDREAMS_EVOLUTIONARY_TEMPLATE"] = EDREAMS_EVOLUTIONARY_TEMPLATE_TEST
os.environ["EDREAMS_PERSONALITY_TEMPLATE"] = EDREAMS_PERSONALITY_TEMPLATE_TEST
os.environ["DREAMS_GEN_TEMPLATE"] = DREAMS_GEN_TEMPLATE_TEST


# 存储
cross_encoder_path = os.environ.get("cross_encoder_path")
embed_model_path =  os.environ.get("embed_model_path") 
start_task_context = os.environ.get("start_task_context")
builder = StructuredTaskStepStoryboard.form_builder(
    llm_runable=llm_with_tools,
    kor_dreams_task_step_llm=kor_dreams_task_step_llm_with_tools,
    start_task_context=start_task_context,
    cross_encoder_path=cross_encoder_path,
    embed_model_path=embed_model_path,
    data_base='search_papers'
)

    
# 初始化任务引擎
allow_init=os.environ.get("allow_init", 'true')
task_engine_builder = builder.loader_task_step_iter_builder(allow_init=allow_init == 'true' )

def worker(step: int, task_engine: TaskEngineBuilder, task_step_store: BaseTaskStepStore, buffer_queue):
    owner = f"step:{step}, task_step_id:{task_engine.task_step_id}, thread {threading.get_native_id()}"
    logger.info(f"{owner}，任务开始")
    try:
        if step & 2 ==0:
            task_engine.llm_runable=guiji_llm
        if not task_engine.check_engine_init():
            task_engine.init_task_engine()
            task_engine.init_task_engine_dreams()
            task_engine.init_task_engine_storyboard_executor()

        logger.info(f"{owner}，storyboard_code_gen_builder")
        code_gen_builder = task_engine.storyboard_code_gen_builder()
        task_step = task_engine.task_step_store.get_task_step(task_engine.task_step_id)
        if task_step.task_step_question_answer is None or len(task_step.task_step_question_answer) == 0:
            task_engine.generate_step_answer(code_gen_builder)
        
        logger.info(f"step:{step}, {owner}，get_mcts_node")
        mcts_node = task_engine.get_mcts_node() 
        mcts_node.initialize()
        logger.info(f"step:{step}, {owner}，get_mcts_node run")
        answer = mcts_node.run()
        
        mcts_node.print()
        print(answer)
        task_step.task_step_question_answer = answer 
        task_step_id = task_engine.task_step_id
        
        task_engine.task_step_store.add_task_step([task_step])
        task_step_store_path = concat_dirs(dirname=f"{builder.base_path}/storage/{task_step_id}", basename=DEFAULT_PERSIST_FNAME)
        task_engine.task_step_store.persist(persist_path=task_step_store_path) 
        
        task_step_store.add_task_step([task_step])
        task_step_store_path = concat_dirs(dirname=f"{builder.base_path}/storage", basename=DEFAULT_PERSIST_FNAME)
        task_step_store.persist(persist_path=task_step_store_path) 

    except Exception as e:
        logger.error("场景加载失败", e)

    logger.info(f"{owner}，任务结束")

    # After completing the task, remove an item from the buffer queue
    buffer_queue.get()
    buffer_queue.task_done()

if __name__ == "__main__":
        
    buffer_queue = queue.Queue(maxsize=6)  # Create the buffer queue with max size of 2
    threads = []
    step = 0
    
    task_step_store = builder.task_step_store
    while not task_engine_builder.empty():
        task_engine = task_engine_builder.get()
        step += 1
        
        # Add a task to the buffer queue to simulate running threads
        buffer_queue.put(1)  # This will block if the buffer is full (i.e., 2 threads are active)
        
        # Create and start a new worker thread
        t = threading.Thread(target=worker,
                                kwargs={"step": step, 
                                        "task_engine": task_engine, 
                                        "task_step_store": task_step_store,
                                        "buffer_queue": buffer_queue},
                                daemon=True)
        t.start()
        threads.append(t)
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
