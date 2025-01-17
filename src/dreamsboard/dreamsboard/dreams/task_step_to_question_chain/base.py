from __future__ import annotations
from abc import ABC
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable
from langchain.chains.base import Chain
from langchain_core.prompts  import PromptTemplate
from langchain.chains import SequentialChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

from dreamsboard.document_loaders import KorLoader

from dreamsboard.engine.storage.storage_context import BaseTaskStepStore

from dreamsboard.document_loaders.protocol.ner_protocol import TaskStepNode
from dreamsboard.dreams.task_step_to_question_chain.prompts import (
    CONVERT_TASK_STEP_TO_QUESTION_PROMPT_TEMPLATE,
    TASK_STEP_QUESTION_TO_GRAPHQL_PROMPT_TEMPLATE
)

from dreamsboard.engine.entity.task_step.task_step import TaskStepContext

from dreamsboard.dreams.task_step_to_question_chain.weaviate.prepare_load import exe_query, get_query_hash

import json
import copy
from sentence_transformers import CrossEncoder
from dreamsboard.engine.storage.task_step_store.types import DEFAULT_PERSIST_FNAME
from dreamsboard.engine.utils import concat_dirs
from typing import List
from dreamsboard.vector.base import CollectionService, DocumentWithVSId
import logging
import os
import re
import pandas as pd
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

PATTERN = re.compile(r"```graphql?([\s\S]*?)```", re.DOTALL)
"""Regex pattern to parse the output."""

class TaskStepToQuestionChain(ABC):
    base_path: str
    start_task_context: str
    task_step_to_question_chain: Chain
    task_step_store: BaseTaskStepStore
    task_step_question_to_graphql_chain: Chain
    collection: CollectionService
    cross_encoder: CrossEncoder
    def __init__(self,
                 base_path: str,
                 start_task_context: str,
                 task_step_store: BaseTaskStepStore,
                 task_step_to_question_chain: Chain,
                 task_step_question_to_graphql_chain: Chain,
                 collection: CollectionService,
                 cross_encoder: CrossEncoder):
        self.base_path = base_path
        self.start_task_context = start_task_context
        self.task_step_store = task_step_store
        self.task_step_to_question_chain = task_step_to_question_chain
        self.task_step_question_to_graphql_chain = task_step_question_to_graphql_chain
        self.collection = collection
        self.cross_encoder = cross_encoder

    @classmethod
    def from_task_step_to_question_chain(
            cls,
            base_path: str,
            start_task_context: str,
            llm_runable: Runnable[LanguageModelInput, BaseMessage],
            task_step_store: BaseTaskStepStore,
            collection: CollectionService,
            cross_encoder: CrossEncoder
    ) -> TaskStepToQuestionChain:
        """

        1、对当前主任务名称创建hash值，作为collection_name
        2、对当前任务步骤名称创建hash值，作为collection_name_context
        """
        prompt_template1 = PromptTemplate(input_variables=["start_task_context",
                                                           "aemo_representation_context",
                                                           "task_step_name",
                                                           "task_step_description",
                                                           "task_step_level"],
                                          template=os.environ.get(
                                              "CONVERT_TASK_STEP_TO_QUESTION_PROMPT_TEMPLATE", CONVERT_TASK_STEP_TO_QUESTION_PROMPT_TEMPLATE
                                          ))

        task_step_to_question_chain = (prompt_template1 | llm_runable | StrOutputParser())

        def wrapper_output(_dict):
            return {
                # 中间变量全部打包输出
                "task_step_question_context": _dict["task_step_question_context"],
            }

        task_step_to_question_chain = ({
                                           "task_step_question_context": task_step_to_question_chain,
                                       }
                                       | RunnableLambda(wrapper_output))

        prompt_template2 = PromptTemplate(input_variables=[
            "collection_name_context",
            "task_step_question"
        ],
            template=os.environ.get("TASK_STEP_QUESTION_TO_GRAPHQL_PROMPT_TEMPLATE", TASK_STEP_QUESTION_TO_GRAPHQL_PROMPT_TEMPLATE)
        )

        task_step_question_to_graphql_chain = (prompt_template2 | llm_runable | StrOutputParser())

        def wrapper_output2(_dict):
            return {
                # 中间变量全部打包输出
                "task_step_question_graphql_context": _dict["task_step_question_graphql_context"],
            }

        task_step_question_to_graphql_chain = ({
                                                   "task_step_question_graphql_context": task_step_question_to_graphql_chain,
                                               }
                                               | RunnableLambda(wrapper_output2))

        return cls(
            base_path=base_path,
            start_task_context=start_task_context,
            task_step_store=task_step_store,
            task_step_to_question_chain=task_step_to_question_chain,
            task_step_question_to_graphql_chain=task_step_question_to_graphql_chain,
            collection=collection,
            cross_encoder=cross_encoder
        )

    def invoke_task_step_to_question(self, task_step_id: str) -> None:
        """
        对开始任务进行抽取，得到任务步骤
        :return:
        """

        task_step_node = self.task_step_store.get_task_step(task_step_id)
        result = self.task_step_to_question_chain.invoke(task_step_node.__dict__)
        task_step_node.task_step_question = result["task_step_question_context"]
        self.task_step_store.add_task_step([task_step_node])

        # 每处理一个任务步骤，就持久化一次
        task_step_store_path = concat_dirs(dirname=f"{self.base_path}/storage/{task_step_id}", basename=DEFAULT_PERSIST_FNAME)
        self.task_step_store.persist(persist_path=task_step_store_path)

    def _insert_into_database(self, union_id_key:str, page_content_key:str, properties_list: List[Dict] = []) -> None:
        """
        插入数据到向量数据库,检查唯一
        :param union_id_key:  唯一标识
        :param properties_list:  数据列表
        :return: None
        """

        union_ids = [str(item.get(union_id_key)) for item in properties_list]

        response = self.collection.get_doc_by_ids(ids=union_ids)

        exist_ids = [o.metadata[union_id_key] for o in response]

        docs = []
        for item in properties_list:
            metadata = {key: value for key, value in item.items() if key != page_content_key}
            if item.get(union_id_key) not in exist_ids:
                doc = DocumentWithVSId(id=item.get(union_id_key), page_content=item.get(page_content_key), metadata=metadata )
                docs.append(doc)

        self.collection.do_add_doc(docs)

    def invoke_task_step_question_context(self, task_step_id: str) -> None:
        """
        对任务步骤进行抽取，得到任务步骤的上下文
        3、增加项目
        """

        task_step_node = self.task_step_store.get_task_step(task_step_id)
        if task_step_node.task_step_question_context is not None and len(task_step_node.task_step_question_context) > 0:
            return

        top_k =100  # 可选参数，默认查询返回前 100个结果
        properties_list = exe_query(task_step_node.task_step_question,top_k)

        # 插入数据到数据库

        self._insert_into_database('ref_id','chunk_text', properties_list)

        # 召回
        response = self.collection.do_search(query=f"{task_step_node.task_step_question}", top_k=10, score_threshold=0.6)

        chunk_texts = []
        ref_ids = []
        chunk_ids = []
        for o in response:
            chunk_texts.append(o.page_content)
            ref_ids.append(o.metadata['ref_id'])
            chunk_ids.append(o.metadata['chunk_id'])

        rankings = self.cross_encoder.rank(
            task_step_node.task_step_question,
            chunk_texts,
            show_progress_bar=True,
            return_documents=True,
            convert_to_tensor=True
        )

        task_step_question_context = []
        # 召回问题前3条，存入task_step_question_context
        for i, ranking in enumerate(rankings[:3]):
            logger.info(f"ref_ids: {ref_ids[i]},chunk_ids: {chunk_ids[i]}, Score: {ranking['score']:.4f}, Text: {ranking['text']}")
            task_step_question_context.append(TaskStepContext(ref_id=str(ref_ids[i]), chunk_id=str(chunk_ids[i]), score=ranking['score'], text=ranking['text']))

        # rankingscorpus_id 的索引与ref_ids的索引一致 
        task_step_node.task_step_question_context = task_step_question_context
        self.task_step_store.add_task_step([task_step_node])

        # 每处理一个任务步骤，就持久化一次
        task_step_store_path = concat_dirs(dirname=f"{self.base_path}/storage/{task_step_id}", basename=DEFAULT_PERSIST_FNAME)
        self.task_step_store.persist(persist_path=task_step_store_path)


    def export_csv_file_path(self, task_step_id: str) -> str:
        """
        3、对召回内容与问题 导出csv文件
        """
        task_step_node = self.task_step_store.get_task_step(task_step_id)

        table_data = []
        row = [
            task_step_id,
            task_step_node.task_step_name,
            task_step_node.task_step_level
        ]

        table_data.append(row)
        row2 = [
            task_step_id,
            task_step_node.task_step_question,
            task_step_node.task_step_level
        ]
        table_data.append(row2)
        for context in task_step_node.task_step_question_context:
            row3 = [
                task_step_id,
                f"ref_ids: {context['ref_id']}, chunk_ids: {context['chunk_id']}, Score: {context['score']:.4f}, Text: {context['text']}",
                task_step_node.task_step_level
            ]
            table_data.append(row3)

        table = pd.DataFrame(table_data, columns=["角色", "内容", "分镜"])

        table.to_csv(f"{self.base_path}/storage/{task_step_id}.csv", index=False, escapechar='\\')

        return f"{self.base_path}/storage/{task_step_id}.csv"