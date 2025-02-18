from __future__ import annotations

import logging
import os
from abc import ABC
from typing import Any, Dict, List

from kor.extraction.parser import KorParser
from kor.nodes import Number, Object, Text
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.base import Chain
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

from dreamsboard.common.csv_data import CSVEncoder
from dreamsboard.document_loaders import KorLoader
from dreamsboard.document_loaders.protocol.ner_protocol import TaskStepNode
from dreamsboard.dreams.aemo_representation_chain.prompts import (
    AEMO_REPRESENTATION_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


class AEMORepresentationChain(ABC):
    aemo_representation_chain: Chain
    kor_dreams_task_step_chain: LLMChain
    kor_schema: Object

    def __init__(
        self,
        start_task_context: str,
        kor_dreams_task_step_chain: LLMChain,
        kor_schema: Object,
        aemo_representation_chain: Chain,
    ):
        self.start_task_context = start_task_context
        self.kor_dreams_task_step_chain = kor_dreams_task_step_chain
        self.kor_schema = kor_schema
        self.aemo_representation_chain = aemo_representation_chain

    @classmethod
    def from_aemo_representation_chain(
        cls,
        llm_runable: Runnable[LanguageModelInput, BaseMessage],
        start_task_context: str,
        kor_dreams_task_step_llm: Runnable[LanguageModelInput, BaseMessage] = None,
    ) -> AEMORepresentationChain:
        # 00-判断情感表征是否符合.txt
        prompt_template1 = PromptTemplate(
            input_variables=["start_task_context"],
            template=os.environ.get(
                "AEMO_REPRESENTATION_PROMPT_TEMPLATE",
                AEMO_REPRESENTATION_PROMPT_TEMPLATE,
            ),
        )

        aemo_representation_chain = prompt_template1 | llm_runable | StrOutputParser()

        def wrapper_output(_dict):
            return {
                # 中间变量全部打包输出
                "aemo_representation_context": _dict["aemo_representation_context"],
            }

        aemo_representation_chain = {
            "aemo_representation_context": aemo_representation_chain,
        } | RunnableLambda(wrapper_output)

        (
            kor_dreams_task_step_chain,
            schema,
        ) = KorLoader.form_kor_dreams_task_step_builder(
            llm_runable=llm_runable
            if kor_dreams_task_step_llm is None
            else kor_dreams_task_step_llm
        )
        return cls(
            start_task_context=start_task_context,
            kor_dreams_task_step_chain=kor_dreams_task_step_chain,
            kor_schema=schema,
            aemo_representation_chain=aemo_representation_chain,
        )

    def invoke_kor_dreams_task_step_context(
        self, aemo_representation_context: str
    ) -> List[TaskStepNode]:
        """
        对开始任务进行抽取，得到任务步骤
        :return:
        """

        response = self.kor_dreams_task_step_chain.run(aemo_representation_context)
        task_step_node_list = []
        if (
            response.get("data") is not None
            and response.get("data").get("script") is not None
        ):
            step_list = response.get("data").get("script")
            for step in step_list:
                task_step_node = TaskStepNode(
                    start_task_context=self.start_task_context,
                    aemo_representation_context=aemo_representation_context,
                    task_step_name=step.get("task_step_name"),
                    task_step_description=step.get("task_step_description"),
                    task_step_level=step.get("task_step_level"),
                )
                task_step_node_list.append(task_step_node)
        else:
            encoder = CSVEncoder(node=self.kor_schema)
            parser = KorParser(encoder=encoder, schema_=self.kor_schema)
            response = parser.parse(response.get("raw"))

            if (
                response.get("data") is not None
                and response.get("data").get("script") is not None
            ):
                step_list = response.get("data").get("script")
                for step in step_list:
                    task_step_node = TaskStepNode(
                        start_task_context=self.start_task_context,
                        aemo_representation_context=aemo_representation_context,
                        task_step_name=step.get("task_step_name"),
                        task_step_description=step.get("task_step_description"),
                        task_step_level=step.get("task_step_level"),
                    )
                    task_step_node_list.append(task_step_node)
        return task_step_node_list

    def invoke_aemo_representation_context(self) -> Dict[str, Any]:
        return self.aemo_representation_chain.invoke(
            {"start_task_context": self.start_task_context}
        )
