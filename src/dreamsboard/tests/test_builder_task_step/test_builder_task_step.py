   
from dreamsboard.engine.storage.task_step_store.simple_task_step_store import SimpleTaskStepStore
from langchain_community.chat_models import ChatOpenAI
from dreamsboard.dreams.builder_task_step.base import StructuredTaskStepStoryboard
from dreamsboard.engine.utils import concat_dirs
from dreamsboard.engine.storage.task_step_store.types import DEFAULT_PERSIST_FNAME
from dreamsboard.common.try_parse_json_object import try_parse_json_object
from dreamsboard.engine.memory.mctsr.prompt import RefineResponse
import logging
import os
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
    os.environ["ZHIPUAI_API_KEY"] = "5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN"
    
    os.environ["OPENAI_API_KEY"] = os.environ.get("ZHIPUAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    kor_dreams_task_step_llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
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
    task_step_store = SimpleTaskStepStore.from_persist_dir("./storage")
    
    cross_encoder_path = "/mnt/ceph/develop/jiawei/model_checkpoint/jina-reranker-v2-base-multilingual"
    builder = StructuredTaskStepStoryboard.form_builder(
        base_path="./",
        llm=llm,
        kor_dreams_task_step_llm=kor_dreams_task_step_llm,
        start_task_context="多模态大模型的技术发展路线是什么样的？", 
        task_step_store=task_step_store,
        cross_encoder_path=cross_encoder_path
    )
    # 初始化任务引擎
    os.environ["OPENAI_API_KEY"] = os.environ.get("ZHIPUAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
    task_engine_builder = builder.loader_task_step_iter_builder(allow_init=True)
    while not task_engine_builder.empty():
        task_engine = task_engine_builder.get()  
        logger.info(task_engine.task_step_id)



def test_builder_task_step_answer():
    os.environ["ZHIPUAI_API_KEY"] = "5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN"
    os.environ["OPENAI_API_KEY"] = os.environ.get("ZHIPUAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    kor_dreams_task_step_llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
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
    task_step_store = SimpleTaskStepStore.from_persist_dir("./storage")
    cross_encoder_path = "/mnt/ceph/develop/jiawei/model_checkpoint/jina-reranker-v2-base-multilingual"
    builder = StructuredTaskStepStoryboard.form_builder(
        base_path="./",
        llm=llm,
        kor_dreams_task_step_llm=kor_dreams_task_step_llm,
        start_task_context="多模态大模型的技术发展路线是什么样的？", 
        task_step_store=task_step_store,
        cross_encoder_path=cross_encoder_path
    )
    # 初始化任务引擎
    os.environ["OPENAI_API_KEY"] = os.environ.get("ZHIPUAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
    task_engine_builder = builder.loader_task_step_iter_builder(allow_init=False)
    step =0
    while not task_engine_builder.empty():
        if step>=2 :
            break
        task_engine = task_engine_builder.get()
        if not task_engine.check_engine_init():
            task_engine.init_task_engine()
            task_engine.init_task_engine_dreams()
            task_engine.init_task_engine_storyboard_executor()

        code_gen_builder = task_engine.storyboard_code_gen_builder()
        task_step = task_engine.task_step_store.get_task_step(task_engine.task_step_id)
        if task_step.task_step_question_answer is None or len(task_step.task_step_question_answer) == 0:
            task_engine.generate_step_answer(code_gen_builder) 
        step +=1


def test_json_parse():
    json_text = """
    {
        "thought": "The thought process behind the answer.",
        "answer": "0.1"
    }
    """
    json_text, json_object = try_parse_json_object(json_text)
    refined_answer = RefineResponse.model_validate_json(
        json_text
    )

def test_builder_task_step_mctsr():
    
    os.environ["ZHIPUAI_API_KEY"] = "5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN"
    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    kor_dreams_task_step_llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key=os.environ.get("ZHIPUAI_API_KEY"),
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
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
    task_step_store = SimpleTaskStepStore.from_persist_dir("./storage")
    cross_encoder_path = "/mnt/ceph/develop/jiawei/model_checkpoint/jina-reranker-v2-base-multilingual"
    builder = StructuredTaskStepStoryboard.form_builder(
        base_path="./",
        llm=llm,
        kor_dreams_task_step_llm=kor_dreams_task_step_llm,
        start_task_context="多模态大模型的技术发展路线是什么样的？", 
        task_step_store=task_step_store,
        cross_encoder_path=cross_encoder_path
    )
    # 初始化任务引擎
    os.environ["OPENAI_API_KEY"] = os.environ.get("ZHIPUAI_API_KEY")
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"
    task_engine_builder = builder.loader_task_step_iter_builder(allow_init=False)
    step =0

    while not task_engine_builder.empty():
        
        if step>=2 :
            break
        task_engine = task_engine_builder.get()
        if not task_engine.check_engine_init():
            task_engine.init_task_engine()
            task_engine.init_task_engine_dreams()
            task_engine.init_task_engine_storyboard_executor()

        try:
            code_gen_builder = task_engine.storyboard_code_gen_builder()
            task_step = task_engine.task_step_store.get_task_step(task_engine.task_step_id)
            if task_step.task_step_question_answer is None or len(task_step.task_step_question_answer) == 0:
                task_engine.generate_step_answer(code_gen_builder)
            mcts_node = task_engine.get_mcts_node()
            answer = mcts_node.run()
            
            mcts_node.print()
            print(answer)
 

        except Exception as e:
            logger.error("场景加载失败", e)

        finally:
            step+=1
