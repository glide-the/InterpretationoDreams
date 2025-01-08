

from dreamsboard.dreams.aemo_representation_chain.base import AEMORepresentationChain
from dreamsboard.engine.entity.task_step.task_step import TaskStepNode
from dreamsboard.engine.storage.task_step_store.simple_task_step_store import SimpleTaskStepStore
 
from dreamsboard.engine.utils import concat_dirs
from langchain_community.chat_models import ChatOpenAI
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
编写符合计算机科学领域的 故事情境提示词，生成研究情境（story_scenario_context），替换现有的langchain会话模板，
1、对这个提示词所要求的输入拆分成子任务， 

"""

def test_aemo_representation_chain_context():
    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key="5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN",
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    ) 
    aemo_representation_chain = AEMORepresentationChain.from_aemo_representation_chain(
            llm=llm,
            start_task_context="有哪些方法可以提升大模型的规划能力，各自优劣是什么？"
        )

    result = aemo_representation_chain.invoke_aemo_representation_context()
    print(result)
    assert result is not None
    assert result.get('aemo_representation_context') is not None


def test_aemo_representation_chain_custom_prompt():
    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key="5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN",
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    ) 
    from tests.test_aemo_representation_chain.prompts import (
        AEMO_REPRESENTATION_PROMPT_TEMPLATE as AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    )
    os.environ["AEMO_REPRESENTATION_PROMPT_TEMPLATE"] = AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    aemo_representation_chain = AEMORepresentationChain.from_aemo_representation_chain(
            llm=llm,
            start_task_context="有哪些方法可以提升大模型的规划能力，各自优劣是什么？"
        )

    result = aemo_representation_chain.invoke_aemo_representation_context()
    print(result)
    assert result is not None
    assert result.get('aemo_representation_context') is not None


def test_aemo_representation_chain_task_step():
    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key="5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN",
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    kor_dreams_task_step_llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key="5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN",
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    ) 
    from tests.test_aemo_representation_chain.prompts import (
        AEMO_REPRESENTATION_PROMPT_TEMPLATE as AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    )
    os.environ["AEMO_REPRESENTATION_PROMPT_TEMPLATE"] = AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    aemo_representation_chain = AEMORepresentationChain.from_aemo_representation_chain(
        llm=llm,
        start_task_context="有哪些方法可以提升大模型的规划能力，各自优劣是什么？",
        kor_dreams_task_step_llm=kor_dreams_task_step_llm
    )

    result = aemo_representation_chain.invoke_aemo_representation_context()
    print(result)
    assert result is not None
    assert result.get('aemo_representation_context') is not None

    task_step_iter = aemo_representation_chain.invoke_kor_dreams_task_step_context(
        aemo_representation_context=result.get('aemo_representation_context')
    )
    print(json.dumps([step.dict() for step in task_step_iter], ensure_ascii=False))
    assert task_step_iter is not None
    assert len(task_step_iter) > 0


def test_aemo_representation_chain_task_step_store():

    llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key="5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN",
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    kor_dreams_task_step_llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4',
        model="glm-4-plus",
        openai_api_key="5fae8f96c5ed49c2b7b21f5c6d74de17.A0bcBERbeZ1gZYoN",
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    ) 
    from tests.test_aemo_representation_chain.prompts import (
        AEMO_REPRESENTATION_PROMPT_TEMPLATE as AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    )
    os.environ["AEMO_REPRESENTATION_PROMPT_TEMPLATE"] = AEMO_REPRESENTATION_PROMPT_TEMPLATE_TEST
    aemo_representation_chain = AEMORepresentationChain.from_aemo_representation_chain(
        llm=llm,
        start_task_context="有哪些方法可以提升大模型的规划能力，各自优劣是什么？",
        kor_dreams_task_step_llm=kor_dreams_task_step_llm
    )

    result = aemo_representation_chain.invoke_aemo_representation_context()
    print(result)
    assert result is not None
    assert result.get('aemo_representation_context') is not None

    task_step_iter = aemo_representation_chain.invoke_kor_dreams_task_step_context(
        aemo_representation_context=result.get('aemo_representation_context')
    )
    print(json.dumps([step.dict() for step in task_step_iter], ensure_ascii=False))
    assert task_step_iter is not None
    assert len(task_step_iter) > 0
    store = SimpleTaskStepStore()
    for step in task_step_iter:
        task_step = TaskStepNode.from_config(cfg={
            "start_task_context": step.start_task_context,
            "aemo_representation_context": step.aemo_representation_context,
            "task_step_name": step.task_step_name,
            "task_step_description": step.task_step_description,
            "task_step_level": step.task_step_level
        })
        
        store.add_task_step([task_step])


    task_step_store_path = concat_dirs(dirname="./storage", basename="task_step_store.json")
    store.persist(persist_path=task_step_store_path)
    assert store is not None