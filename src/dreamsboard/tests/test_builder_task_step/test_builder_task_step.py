import logging
import multiprocessing
import os
import queue

from langchain_community.chat_models import ChatOpenAI

from dreamsboard.common import _get_assistants_tool
from dreamsboard.common.try_parse_json_object import try_parse_json_object
from dreamsboard.dreams.builder_task_step.base import StructuredTaskStepStoryboard
from dreamsboard.dreams.task_step_md.base import TaskStepMD
from dreamsboard.dreams.task_step_to_question_chain.weaviate.prepare_load import (
    get_query_hash,
)
from dreamsboard.engine.memory.mctsr.prompt import RefineResponse
from dreamsboard.engine.storage.task_step_store.simple_task_step_store import (
    SimpleTaskStepStore,
)
from dreamsboard.engine.storage.task_step_store.types import (
    DEFAULT_PERSIST_FNAME,
    BaseTaskStepStore,
)
from dreamsboard.engine.task_engine_builder.core import TaskEngineBuilder
from dreamsboard.engine.utils import concat_dirs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)

"""

#### 场景加载模块

编写符合计算机科学领域的 故事情境提示词，生成研究情境（story_scenario_context），替换现有的langchain会话模板，
1、对这个提示词所要求的输入拆分成子任务，
2、对每个子任务指令转换为子问题，召回问题前3条，
3、对召回内容与问题 导出csv文件，合成会话内容变量（scene_content）

	对每个子问题相关的召回内容，转换为第一人称的会话总结（研究场景（scene_monologue_context）），

	1、对召回内容与问题拼接，对每个前3条组成一个总结任务的提示词，为每个任务标记唯一编号，组成任务上下文（story_board_summary_context）
	2、加载编号和story_board_summary_context，转换会话信息
"""


def test_builder_task_step():
    llm = ChatOpenAI(
        openai_api_base=os.environ.get("API_BASE"),
        model=os.environ.get("API_MODEL"),
        openai_api_key=os.environ.get("API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    kor_dreams_task_step_llm = ChatOpenAI(
        openai_api_base=os.environ.get("API_BASE"),
        model=os.environ.get("API_MODEL"),
        openai_api_key=os.environ.get("API_KEY"),
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )

    if "glm" in os.environ.get("API_MODEL"):
        tools = [
            {
                "type": "web_search",
                "web_search": {"enable": False, "search_result": False},
            }
        ]
    else:
        tools = []
    llm_with_tools = llm.bind(tools=[_get_assistants_tool(tool) for tool in tools])
    kor_dreams_task_step_llm_with_tools = kor_dreams_task_step_llm.bind(
        tools=[_get_assistants_tool(tool) for tool in tools]
    )

    from tests.test_builder_task_step.prompts import (
        AEMO_REPRESENTATION_PROMPT_TEMPLATE as AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        DREAMS_GEN_TEMPLATE as DREAMS_GEN_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        EDREAMS_EVOLUTIONARY_TEMPLATE as EDREAMS_EVOLUTIONARY_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        EDREAMS_PERSONALITY_TEMPLATE as EDREAMS_PERSONALITY_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        STORY_BOARD_SCENE_TEMPLATE as STORY_BOARD_SCENE_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE as STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST,
    )

    os.environ[
        "AEMO_REPRESENTATION_PROMPT_TEMPLATE"
    ] = AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    os.environ["STORY_BOARD_SCENE_TEMPLATE"] = STORY_BOARD_SCENE_TEMPLATE_TEST
    os.environ[
        "STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE"
    ] = STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST
    os.environ["EDREAMS_EVOLUTIONARY_TEMPLATE"] = EDREAMS_EVOLUTIONARY_TEMPLATE_TEST
    os.environ["EDREAMS_PERSONALITY_TEMPLATE"] = EDREAMS_PERSONALITY_TEMPLATE_TEST
    os.environ["DREAMS_GEN_TEMPLATE"] = DREAMS_GEN_TEMPLATE_TEST

    cross_encoder_path = (
        "/mnt/ceph/develop/jiawei/model_checkpoint/jina-reranker-v2-base-multilingual"
    )
    embed_model_path = "/mnt/ceph/develop/jiawei/model_checkpoint/m3e-base"
    start_task_context = "什么是损失函数？"
    builder = StructuredTaskStepStoryboard.form_builder(
        llm_runable=llm_with_tools,
        kor_dreams_task_step_llm=kor_dreams_task_step_llm_with_tools,
        start_task_context=start_task_context,
        cross_encoder_path=cross_encoder_path,
        embed_model_path=embed_model_path,
    )
    # 初始化任务引擎
    task_engine_builder = builder.loader_task_step_iter_builder(allow_init=True)
    while not task_engine_builder.empty():
        task_engine = task_engine_builder.get()
        logger.info(task_engine.task_step_id)

    assert builder.base_path == f"./{get_query_hash(start_task_context)}/"


def test_builder_task_step_answer():
    llm = ChatOpenAI(
        openai_api_base=os.environ.get("API_BASE"),
        model=os.environ.get("API_MODEL"),
        openai_api_key=os.environ.get("API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    kor_dreams_task_step_llm = ChatOpenAI(
        openai_api_base=os.environ.get("API_BASE"),
        model=os.environ.get("API_MODEL"),
        openai_api_key=os.environ.get("API_KEY"),
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )

    if "glm" in os.environ.get("API_MODEL"):
        tools = [
            {
                "type": "web_search",
                "web_search": {"enable": False, "search_result": False},
            }
        ]
    else:
        tools = []
    llm_with_tools = llm.bind(tools=[_get_assistants_tool(tool) for tool in tools])
    kor_dreams_task_step_llm_with_tools = kor_dreams_task_step_llm.bind(
        tools=[_get_assistants_tool(tool) for tool in tools]
    )

    from tests.test_builder_task_step.prompts import (
        AEMO_REPRESENTATION_PROMPT_TEMPLATE as AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        DREAMS_GEN_TEMPLATE as DREAMS_GEN_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        EDREAMS_EVOLUTIONARY_TEMPLATE as EDREAMS_EVOLUTIONARY_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        EDREAMS_PERSONALITY_TEMPLATE as EDREAMS_PERSONALITY_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        STORY_BOARD_SCENE_TEMPLATE as STORY_BOARD_SCENE_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE as STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST,
    )

    os.environ[
        "AEMO_REPRESENTATION_PROMPT_TEMPLATE"
    ] = AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    os.environ["STORY_BOARD_SCENE_TEMPLATE"] = STORY_BOARD_SCENE_TEMPLATE_TEST
    os.environ[
        "STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE"
    ] = STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST
    os.environ["EDREAMS_EVOLUTIONARY_TEMPLATE"] = EDREAMS_EVOLUTIONARY_TEMPLATE_TEST
    os.environ["EDREAMS_PERSONALITY_TEMPLATE"] = EDREAMS_PERSONALITY_TEMPLATE_TEST
    os.environ["DREAMS_GEN_TEMPLATE"] = DREAMS_GEN_TEMPLATE_TEST

    # 存储
    cross_encoder_path = (
        "/mnt/ceph/develop/jiawei/model_checkpoint/jina-reranker-v2-base-multilingual"
    )
    embed_model_path = "/mnt/ceph/develop/jiawei/model_checkpoint/m3e-base"
    start_task_context = "什么是损失函数？"
    builder = StructuredTaskStepStoryboard.form_builder(
        llm_runable=llm_with_tools,
        kor_dreams_task_step_llm=kor_dreams_task_step_llm_with_tools,
        start_task_context=start_task_context,
        cross_encoder_path=cross_encoder_path,
        embed_model_path=embed_model_path,
    )
    # 初始化任务引擎
    os.environ["OPENAI_API_KEY"] = os.environ.get("ZHIPUAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
    task_engine_builder = builder.loader_task_step_iter_builder(allow_init=False)
    task_step_store = builder.task_step_store
    while not task_engine_builder.empty():
        task_engine = task_engine_builder.get()
        if not task_engine.check_engine_init():
            task_engine.init_task_engine()
            task_engine.init_task_engine_dreams()
            task_engine.init_task_engine_storyboard_executor()

        code_gen_builder = task_engine.storyboard_code_gen_builder()
        task_step = task_engine.task_step_store.get_task_step(task_engine.task_step_id)
        if (
            task_step.task_step_question_answer is None
            or len(task_step.task_step_question_answer) == 0
        ):
            task_engine.generate_step_answer(code_gen_builder)


def test_json_parse():
    json_text = """
    {
        "thought": "The thought process behind the answer.",
        "answer": "0.1"
    }
    """
    json_text, json_object = try_parse_json_object(json_text)
    refined_answer = RefineResponse.model_validate_json(json_text)


def test_builder_task_step_mctsr():
    llm = ChatOpenAI(
        openai_api_base=os.environ.get("API_BASE"),
        model=os.environ.get("API_MODEL"),
        openai_api_key=os.environ.get("API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    kor_dreams_task_step_llm = ChatOpenAI(
        openai_api_base=os.environ.get("API_BASE"),
        model=os.environ.get("API_MODEL"),
        openai_api_key=os.environ.get("API_KEY"),
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )

    if "glm" in os.environ.get("API_MODEL"):
        tools = [
            {
                "type": "web_search",
                "web_search": {"enable": False, "search_result": False},
            }
        ]
    else:
        tools = []
    llm_with_tools = llm.bind(tools=[_get_assistants_tool(tool) for tool in tools])
    kor_dreams_task_step_llm_with_tools = kor_dreams_task_step_llm.bind(
        tools=[_get_assistants_tool(tool) for tool in tools]
    )

    from tests.test_builder_task_step.prompts import (
        AEMO_REPRESENTATION_PROMPT_TEMPLATE as AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        DREAMS_GEN_TEMPLATE as DREAMS_GEN_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        EDREAMS_EVOLUTIONARY_TEMPLATE as EDREAMS_EVOLUTIONARY_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        EDREAMS_PERSONALITY_TEMPLATE as EDREAMS_PERSONALITY_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        STORY_BOARD_SCENE_TEMPLATE as STORY_BOARD_SCENE_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE as STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST,
    )

    os.environ[
        "AEMO_REPRESENTATION_PROMPT_TEMPLATE"
    ] = AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    os.environ["STORY_BOARD_SCENE_TEMPLATE"] = STORY_BOARD_SCENE_TEMPLATE_TEST
    os.environ[
        "STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE"
    ] = STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST
    os.environ["EDREAMS_EVOLUTIONARY_TEMPLATE"] = EDREAMS_EVOLUTIONARY_TEMPLATE_TEST
    os.environ["EDREAMS_PERSONALITY_TEMPLATE"] = EDREAMS_PERSONALITY_TEMPLATE_TEST
    os.environ["DREAMS_GEN_TEMPLATE"] = DREAMS_GEN_TEMPLATE_TEST

    # 存储
    cross_encoder_path = (
        "/mnt/ceph/develop/jiawei/model_checkpoint/jina-reranker-v2-base-multilingual"
    )
    embed_model_path = "/mnt/ceph/develop/jiawei/model_checkpoint/m3e-base"
    start_task_context = "什么是损失函数？"
    builder = StructuredTaskStepStoryboard.form_builder(
        llm_runable=llm_with_tools,
        kor_dreams_task_step_llm=kor_dreams_task_step_llm_with_tools,
        start_task_context=start_task_context,
        cross_encoder_path=cross_encoder_path,
        embed_model_path=embed_model_path,
    )
    # 初始化任务引擎
    task_engine_builder = builder.loader_task_step_iter_builder(allow_init=False)
    # step =0
    task_step_store = builder.task_step_store
    while not task_engine_builder.empty():
        task_engine = task_engine_builder.get()
        # step+=1
        # if step<=7 :
        #     continue
        if not task_engine.check_engine_init():
            task_engine.init_task_engine()
            task_engine.init_task_engine_dreams()
            task_engine.init_task_engine_storyboard_executor()

        try:
            code_gen_builder = task_engine.storyboard_code_gen_builder()
            task_step = task_engine.task_step_store.get_task_step(
                task_engine.task_step_id
            )
            if (
                task_step.task_step_question_answer is None
                or len(task_step.task_step_question_answer) == 0
            ):
                task_engine.generate_step_answer(code_gen_builder)
            mcts_node = task_engine.get_mcts_node()
            answer = mcts_node.run()

            mcts_node.print()
            print(answer)
            task_step.task_step_question_answer = answer
            task_step_id = task_engine.task_step_id

            task_engine.task_step_store.add_task_step([task_step])
            task_step_store_path = concat_dirs(
                dirname=f"{builder.base_path}/storage/{task_step_id}",
                basename=DEFAULT_PERSIST_FNAME,
            )
            task_engine.task_step_store.persist(persist_path=task_step_store_path)

            task_step_store.add_task_step([task_step])
            task_step_store_path = concat_dirs(
                dirname=f"{builder.base_path}/storage", basename=DEFAULT_PERSIST_FNAME
            )
            task_step_store.persist(persist_path=task_step_store_path)

        except Exception as e:
            logger.error("场景加载失败", e)


def test_task_step_md():
    task_step_store = SimpleTaskStepStore.from_persist_dir(
        "/mnt/ceph/develop/jiawei/InterpretationoDreams/906079e48187e1dadf611f8c2e9afabb/storage"
    )
    task_step_md = TaskStepMD(task_step_store)
    md_text = task_step_md.format_md()
    print(md_text.text)


def test_builder_task_step_mctsr_threads(setup_log):
    import threading

    # llm = ChatOpenAI(
    #     openai_api_base=os.environ.get("API_BASE"),
    #     model=os.environ.get("API_MODEL"),
    #     openai_api_key=os.environ.get("API_KEY"),
    #     verbose=True,
    #     temperature=0.1,
    #     top_p=0.9,
    # )
    # kor_dreams_task_step_llm = ChatOpenAI(
    #     openai_api_base=os.environ.get("API_BASE"),
    #     model=os.environ.get("API_MODEL"),
    #     openai_api_key=os.environ.get("API_KEY"),
    #     verbose=True,
    #     temperature=0.95,
    #     top_p=0.70,
    # )
    # if 'glm' in os.environ.get("API_MODEL"):

    #     tools= [ { "type": "web_search",   "web_search": {"enable": False ,"search_result": False   }}]
    # else:
    #     tools = []
    # llm_with_tools = llm.bind(   tools=[_get_assistants_tool(tool) for tool in tools] )
    # kor_dreams_task_step_llm_with_tools = kor_dreams_task_step_llm.bind(   tools=[_get_assistants_tool(tool) for tool in tools] )

    llm = ChatOpenAI(
        openai_api_base=os.environ.get("KIMI_API_BASE"),
        model=os.environ.get("KIMI_API_MODEL"),
        openai_api_key=os.environ.get("KIMI_API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
        max_tokens=8192,
    )

    guiji_llm = ChatOpenAI(
        openai_api_base=os.environ.get("PPINFRA_API_BASE"),
        model=os.environ.get("PPINFRA_API_MODEL"),
        openai_api_key=os.environ.get("PPINFRA_API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
        max_tokens=8192,
    )
    llm_with_tools = llm
    kor_dreams_task_step_llm_with_tools = guiji_llm

    from tests.test_builder_task_step.prompts import (
        AEMO_REPRESENTATION_PROMPT_TEMPLATE as AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        DREAMS_GEN_TEMPLATE as DREAMS_GEN_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        EDREAMS_EVOLUTIONARY_TEMPLATE as EDREAMS_EVOLUTIONARY_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        EDREAMS_PERSONALITY_TEMPLATE as EDREAMS_PERSONALITY_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        STORY_BOARD_SCENE_TEMPLATE as STORY_BOARD_SCENE_TEMPLATE_TEST,
    )
    from tests.test_builder_task_step.prompts import (
        STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE as STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST,
    )

    os.environ[
        "AEMO_REPRESENTATION_PROMPT_TEMPLATE"
    ] = AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    os.environ["STORY_BOARD_SCENE_TEMPLATE"] = STORY_BOARD_SCENE_TEMPLATE_TEST
    os.environ[
        "STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE"
    ] = STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE_TEST
    os.environ["EDREAMS_EVOLUTIONARY_TEMPLATE"] = EDREAMS_EVOLUTIONARY_TEMPLATE_TEST
    os.environ["EDREAMS_PERSONALITY_TEMPLATE"] = EDREAMS_PERSONALITY_TEMPLATE_TEST
    os.environ["DREAMS_GEN_TEMPLATE"] = DREAMS_GEN_TEMPLATE_TEST

    # 存储
    cross_encoder_path = (
        "/mnt/ceph/develop/jiawei/model_checkpoint/jina-reranker-v2-base-multilingual"
    )
    embed_model_path = "/mnt/ceph/develop/jiawei/model_checkpoint/m3e-base"
    start_task_context = "MCTS在PRM偏好策略模型微调的应用探索综述"
    builder = StructuredTaskStepStoryboard.form_builder(
        llm_runable=llm,
        kor_dreams_task_step_llm=kor_dreams_task_step_llm_with_tools,
        start_task_context=start_task_context,
        cross_encoder_path=cross_encoder_path,
        embed_model_path=embed_model_path,
    )

    # 初始化任务引擎
    task_engine_builder = builder.loader_task_step_iter_builder(allow_init=False)

    def worker(
        step: int,
        task_engine: TaskEngineBuilder,
        task_step_store: BaseTaskStepStore,
        buffer_queue,
    ):
        owner = f"step:{step}, task_step_id:{task_engine.task_step_id}, thread {threading.get_native_id()}"
        logger.info(f"{owner}，任务开始")
        try:
           
            task_engine.llm_runable = llm
            if not task_engine.check_engine_init():
                task_engine.init_task_engine()
                task_engine.init_task_engine_dreams()
                task_engine.init_task_engine_storyboard_executor()

            logger.info(f"{owner}，storyboard_code_gen_builder")
            code_gen_builder = task_engine.storyboard_code_gen_builder()
            task_step = task_engine.task_step_store.get_task_step(
                task_engine.task_step_id
            )
            if (
                task_step.task_step_question_answer is None
                or len(task_step.task_step_question_answer) == 0
            ):
                task_engine.generate_step_answer(code_gen_builder)

            logger.info(f"step:{step}, {owner}，get_mcts_node")
            mcts_node = task_engine.get_mcts_node()

            logger.info(f"step:{step}, {owner}，get_mcts_node run")
            mcts_node.initialize()
            answer = mcts_node.run()

            mcts_node.print()
            print(answer)
            task_step.task_step_question_answer = answer
            task_step_id = task_engine.task_step_id

            task_engine.task_step_store.add_task_step([task_step])
            task_step_store_path = concat_dirs(
                dirname=f"{builder.base_path}/storage/{task_step_id}",
                basename=DEFAULT_PERSIST_FNAME,
            )
            task_engine.task_step_store.persist(persist_path=task_step_store_path)

            task_step_store.add_task_step([task_step])
            task_step_store_path = concat_dirs(
                dirname=f"{builder.base_path}/storage", basename=DEFAULT_PERSIST_FNAME
            )
            task_step_store.persist(persist_path=task_step_store_path)

        except Exception as e:
            logger.error("场景加载失败", e)

        logger.info(f"{owner}，任务结束")
        # After completing the task, remove an item from the buffer queue
        buffer_queue.get()
        buffer_queue.task_done()

    buffer_queue = queue.Queue(maxsize=2)  # Create the buffer queue with max size of 2
    threads = []
    step = 0

    task_step_store = builder.task_step_store
    while not task_engine_builder.empty():
        task_engine = task_engine_builder.get()
        step += 1

        # Add a task to the buffer queue to simulate running threads
        buffer_queue.put(
            1
        )  # This will block if the buffer is full (i.e., 2 threads are active)

        # Create and start a new worker thread
        t = threading.Thread(
            target=worker,
            kwargs={
                "step": step,
                "task_engine": task_engine,
                "task_step_store": task_step_store,
                "buffer_queue": buffer_queue,
            },
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()
