 
  
from dreamsboard.dreams.task_step_to_question_chain.base import TaskStepToQuestionChain
from dreamsboard.engine.storage.task_step_store.simple_task_step_store import SimpleTaskStepStore
from langchain_community.chat_models import ChatOpenAI
from dreamsboard.engine.utils import concat_dirs
from dreamsboard.document_loaders.csv_structured_storyboard_loader import StructuredStoryboardCSVBuilder

from sentence_transformers import CrossEncoder
import logging
import os
import json
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)

"""
#### 场景加载模块

编写符合计算机科学领域的 故事情境提示词，生成研究情境（story_scenario_context），替换现有的langchain会话模板，
对每个子任务指令转换为子问题
召回问题前3条,存入task_step_question_context
调用llm，生成task_step_question_answer
"""


def test_invoke_task_step_to_question():
    

    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key="testkey",
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )   
    task_step_store = SimpleTaskStepStore.from_persist_dir(persist_dir="./storage")
    
    os.environ["ZHIPUAI_API_KEY"] = "testkey"


    cross_encoder_path = "D:\model\jina-reranker-v2-base-multilingual"

    collection = FaissCollectionService(
        kb_name="faiss",
        embed_model="D:\model\m3e-base",
        vector_name="samples",
        device="cpu"
    )
    task_step_to_question_chain = TaskStepToQuestionChain.from_task_step_to_question_chain(
        base_path="./",
        llm=llm, 
        task_step_store=task_step_store,
        collection=collection,
        cross_encoder_path=cross_encoder_path
    )
    
    task_step_id = list(task_step_store.task_step_all.keys())[0]
    task_step_to_question_chain.invoke_task_step_to_question(task_step_id)
    assert task_step_store.task_step_all is not None


def test_invoke_task_step_question_context():
    

    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key="testkey",
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )   
    os.environ["ZHIPUAI_API_KEY"] = "testkey"
    task_step_store = SimpleTaskStepStore.from_persist_dir(persist_dir="./storage")


    cross_encoder_path = "D:\model\jina-reranker-v2-base-multilingual"

    collection = FaissCollectionService(
        kb_name="faiss",
        embed_model="D:\model\m3e-base",
        vector_name="samples",
        device="cpu"
    )
    task_step_to_question_chain = TaskStepToQuestionChain.from_task_step_to_question_chain(
        base_path="./",
        llm=llm,
        task_step_store=task_step_store,
        collection=collection,
        cross_encoder_path=cross_encoder_path
    )

    task_step_id = list(task_step_store.task_step_all.keys())[0]
    task_step_to_question_chain.invoke_task_step_question_context(task_step_id)
    assert task_step_store.task_step_all is not None


def test_export_csv_file_path():
    

    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key="testkey",
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )   
    os.environ["ZHIPUAI_API_KEY"] = "testkey"
    task_step_store = SimpleTaskStepStore.from_persist_dir(persist_dir="./storage")


    cross_encoder_path = "D:\model\jina-reranker-v2-base-multilingual"

    collection = FaissCollectionService(
        kb_name="faiss",
        embed_model="D:\model\m3e-base",
        vector_name="samples",
        device="cpu"
    )
    task_step_to_question_chain = TaskStepToQuestionChain.from_task_step_to_question_chain(
        base_path="./",
        llm=llm,
        task_step_store=task_step_store,
        collection=collection,
        cross_encoder_path=cross_encoder_path
    )

    task_step_id = list(task_step_store.task_step_all.keys())[0]
    csv_file_path = task_step_to_question_chain.export_csv_file_path(task_step_id)
    logger.info("csv_file_path:" + csv_file_path)
    assert csv_file_path is not None

    builder = StructuredStoryboardCSVBuilder(
        csv_file_path=csv_file_path
    )
    builder.load()  # 替换为你的CSV文件路径
    selected_columns = ["story_board_role", "story_board_text", "story_board"]
    formatted_text = builder.build_text(task_step_id, selected_columns)
    logger.info("formatted_text:" + formatted_text)
    assert formatted_text is not None