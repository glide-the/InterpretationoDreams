from __future__ import annotations
from abc import ABC
from typing import Any, Dict
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.prompts  import PromptTemplate
from langchain.chains import SequentialChain
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

from dreamsboard.document_loaders import KorLoader

from dreamsboard.engine.storage.storage_context import BaseTaskStepStore

from dreamsboard.document_loaders.protocol.ner_protocol import TaskStepNode
from dreamsboard.dreams.task_step_to_question_chain.prompts import (
    CONVERT_TASK_STEP_TO_QUESTION_PROMPT_TEMPLATE,
)
from typing import List
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)
"""
#### 场景加载模块

编写符合计算机科学领域的 故事情境提示词，生成研究情境（story_scenario_context），替换现有的langchain会话模板，
对每个子任务指令转换为子问题
召回问题前3条,存入task_step_question_context
调用llm，生成task_step_question_answer
"""

class TaskStepToQuestionChain(ABC):
    task_step_to_question_chain: Chain
    task_step_store: BaseTaskStepStore
    def __init__(self,
                 task_step_store: BaseTaskStepStore,
                 task_step_to_question_chain: Chain):
        self.task_step_store = task_step_store
        self.task_step_to_question_chain = task_step_to_question_chain
    
    @classmethod
    def from_task_step_to_question_chain(
            cls,
            llm: BaseLanguageModel,
            task_step_store: BaseTaskStepStore
    ) -> TaskStepToQuestionChain:
        prompt_template1 = PromptTemplate(input_variables=["start_task_context", 
                                                           "aemo_representation_context",
                                                           "task_step_name",
                                                           "task_step_description",
                                                           "task_step_level"],
                                          template=os.environ.get(
                                "CONVERT_TASK_STEP_TO_QUESTION_PROMPT_TEMPLATE", CONVERT_TASK_STEP_TO_QUESTION_PROMPT_TEMPLATE
                            ))

        task_step_to_question_chain = (prompt_template1 | llm | StrOutputParser())
 
        def wrapper_output(_dict):
            return {
                # 中间变量全部打包输出
                "task_step_question_context": _dict["task_step_question_context"],
            }

        task_step_to_question_chain = ({
                                        "task_step_question_context": task_step_to_question_chain,
                                    }
                                    | RunnableLambda(wrapper_output))

 
        return cls(task_step_store=task_step_store,
                   task_step_to_question_chain=task_step_to_question_chain)

    def invoke_task_step_to_question(self) -> None:
        """
        对开始任务进行抽取，得到任务步骤
        :return:
        """ 

        task_step_all = self.task_step_store.task_step_all
        for task_step_id, task_step_node in task_step_all.items():
            result = self.task_step_to_question_chain.invoke(task_step_node.__dict__)
            task_step_node.task_step_question = result["task_step_question_context"]
            self.task_step_store.add_task_step([task_step_node])
            # 每处理一个任务步骤，就持久化一次
            self.task_step_store.persist()

    def invoke_task_step_to_question_context(self) -> None:
        pass