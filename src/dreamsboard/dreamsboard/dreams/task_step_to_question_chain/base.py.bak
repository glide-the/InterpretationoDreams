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

from dreamsboard.dreams.task_step_to_question_chain.weaviate.context_collections import init_context_collections
from dreamsboard.dreams.task_step_to_question_chain.weaviate.prepare_load import exe_query, get_query_hash
from dreamsboard.dreams.task_step_to_question_chain.weaviate.prepare_load import insert_into_database
from dreamsboard.dreams.task_step_to_question_chain.weaviate.init_networkx_concept import find_root_nodes
from dreamsboard.dreams.task_step_to_question_chain.weaviate.init_networkx_concept import find_high_outdegree_concepts
from dreamsboard.dreams.task_step_to_question_chain.weaviate.init_networkx_concept import find_high_indegree_nodes
from dreamsboard.dreams.task_step_to_question_chain.weaviate.init_networkx_concept import create_interactive_graph
from dreamsboard.dreams.task_step_to_question_chain.weaviate.init_networkx_concept import find_simple_path
from dreamsboard.dreams.task_step_to_question_chain.weaviate.init_networkx_concept import create_G

import json
import copy
from sentence_transformers import CrossEncoder
from weaviate.client import WeaviateClient
from weaviate.classes.query import Filter, Rerank, MetadataQuery
from dreamsboard.engine.storage.task_step_store.types import DEFAULT_PERSIST_FNAME
from dreamsboard.engine.utils import concat_dirs
from typing import List
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
    collection_name: str
    collection_name_context: str
    task_step_to_question_chain: Chain
    task_step_store: BaseTaskStepStore
    task_step_question_to_graphql_chain: Chain
    client: WeaviateClient
    cross_encoder: CrossEncoder
    def __init__(self,
                 base_path: str,
                 start_task_context: str,
                 collection_name: str,
                 collection_name_context: str,
                 task_step_store: BaseTaskStepStore,
                 task_step_to_question_chain: Chain,
                 task_step_question_to_graphql_chain: Chain,
                 client: WeaviateClient,
                 cross_encoder: CrossEncoder):
        self.base_path = base_path
        self.start_task_context = start_task_context
        self.collection_name = collection_name
        self.collection_name_context = collection_name_context
        self.task_step_store = task_step_store
        self.task_step_to_question_chain = task_step_to_question_chain
        self.task_step_question_to_graphql_chain = task_step_question_to_graphql_chain
        self.client = client
        self.cross_encoder = cross_encoder

    @classmethod
    def from_task_step_to_question_chain(
            cls,
            base_path: str,
            start_task_context: str,
            llm_runable: Runnable[LanguageModelInput, BaseMessage],
            task_step_store: BaseTaskStepStore,
            client: WeaviateClient,
            cross_encoder_path: str
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

        cross_encoder = CrossEncoder(
            cross_encoder_path,
            automodel_args={"torch_dtype": "auto"},
            trust_remote_code=True,
        )
        
           
        collection_id = get_query_hash(start_task_context)
        # TODO 之后单独设计召回模块，由模块生成collection_name, collection_name_context
        collection_name, collection_name_context = init_context_collections(client, collection_id)
        return cls(
            base_path=base_path,
            start_task_context=start_task_context,
            collection_name=collection_name,
            collection_name_context=collection_name_context,
            task_step_store=task_step_store,
            task_step_to_question_chain=task_step_to_question_chain,
            task_step_question_to_graphql_chain=task_step_question_to_graphql_chain,
            client=client,
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


    def _retrieve_task_step_question_context(self, task_step_id: str, collection_name_context: str, task_step_node: TaskStepNode) -> tuple[List[str], str, str]:
        """基于概念路径检索方法"""

        result = self.task_step_question_to_graphql_chain.invoke({
            "collection_name_context": collection_name_context,
            "task_step_question": task_step_node.task_step_question
        })
        # 解析出graphql查询语句
        action_match = PATTERN.search(result["task_step_question_graphql_context"])
        if action_match is not None:
            # 去掉前后空格
            temp_graphql_context = action_match.group(1).strip()
            # 去掉 ```graphql , 后```
            task_step_question_graphql_context = temp_graphql_context.replace("```graphql", "").replace("```", "")
        else:
            raise ValueError("No graphql query found in the response")

        response = self.client.graphql_raw_query(task_step_question_graphql_context)
        data_list = response.get[collection_name_context]
        data_list_copy = copy.deepcopy(data_list)
                
        # 找出根节点
        root_nodes = find_root_nodes(data_list_copy)
        logger.info(f"Root nodes:{json.dumps(root_nodes)}")

        # 找出出度大于3的节点
        high_outdegree_concepts = find_high_outdegree_concepts(data_list_copy)
        logger.info(f"Concepts with more than 3 outgoing edges:{high_outdegree_concepts}")


        # 找出入度大于3的节点
        high_indegree_concepts = find_high_indegree_nodes(data_list_copy)
        logger.info(f"Concepts with more than 3 going edges:{high_indegree_concepts}")


        create_interactive_graph(
            data_list_copy,
            root_nodes_with_concept=root_nodes,
            high_outdegree_concepts=high_outdegree_concepts,
            high_indegree_concepts=high_indegree_concepts,
            filename=f"{self.base_path}/storage/{task_step_id}_semantic_path_interactive.html"
        )

        G = create_G(data_list_copy,  root_nodes_with_concept=root_nodes,
            high_outdegree_concepts=high_outdegree_concepts,
            high_indegree_concepts=high_indegree_concepts)

        result_paths = find_simple_path(G,  root_nodes_with_concept=root_nodes,
            high_outdegree_concepts=high_outdegree_concepts)

        logger.info(f"Root nodes:{json.dumps(root_nodes)}")
        
        logger.info(f"Concepts with more than 3 outgoing edges:{high_outdegree_concepts}")
        
        logger.info(f"Concepts with more than 3 going edges:{high_indegree_concepts}")

        # result_paths 中即为符合条件的路径列表
        refIds = []
        if len(result_paths) == 0:
            raise ValueError("No paths found")
        # 输出路径以及路径中节点的refId信息
        for p in result_paths:
            logger.info(f"Path: " + " -> ".join(p))
            # 输出路径中每个节点的refId字段信息
            for node in p:
                logger.info(f"Node: {node}, refIds: {G.nodes[node].get('refId')}")
                refIds.extend(G.nodes[node].get('refId'))

        return refIds

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
        insert_into_database(self.client, self.collection_name, 'ref_id', properties_list)

        context_properties_list = [{
            "refId": item.get('ref_id'),
            "paperId": item.get('paper_id'),
            "chunkId": item.get('chunk_id'),
            "chunkText": item.get('chunk_text')
        } for item in properties_list]

        insert_into_database(self.client, self.collection_name_context, 'refId', context_properties_list)
        retrieve_ids = []
        try:
            retrieve_ids = self._retrieve_task_step_question_context(task_step_id, self.collection_name_context, task_step_node)
        except Exception as e:
            logger.error(f"Error retrieving task step question context: {e}")
            
        if len(retrieve_ids) == 0:
            
            collection = self.client.collections.get(self.collection_name) 

            # vector_names = ["chunk_text"]
            response = collection.query.hybrid(
                query=task_step_node.task_step_question,  
                # target_vector=vector_names,  # Specify the target vector for named vector collections
                limit=1,
                alpha=0.1,
                query_properties=["chunk_text"],  
                return_metadata=MetadataQuery(score=True, explain_score=True, distance=True),
            )
        else:
            # 对refIds进行编码
            collection = self.client.collections.get(self.collection_name)
            response = collection.query.fetch_objects(
                filters=Filter.by_property("ref_id").contains_any(retrieve_ids), 
                return_metadata=MetadataQuery(score=True, explain_score=True, distance=True),
            )

        chunk_texts = []
        ref_ids = []
        for o in response.objects:  
            chunk_texts.append(o.properties['chunk_text'])
            ref_ids.append(o.properties['ref_id'])
        
        rankings = self.cross_encoder.rank(task_step_node.task_step_question, chunk_texts, return_documents=True, convert_to_tensor=True)
                    
        task_step_question_context = []
        # 召回问题前3条，存入task_step_question_context
        for i, ranking in enumerate(rankings[:3]):
            logger.info(f"ref_ids: {ref_ids[i]}, Score: {ranking['score']:.4f}, Text: {ranking['text']}")
            task_step_question_context.append(TaskStepContext(ref_id=ref_ids[i], score=ranking['score'], text=ranking['text']))

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
                f"ref_ids: {context['ref_id']}, Score: {context['score']:.4f}, Text: {context['text']}",
                task_step_node.task_step_level
            ]
            table_data.append(row3)

        table = pd.DataFrame(table_data, columns=["角色", "内容", "分镜"])

        table.to_csv(f"{self.base_path}/storage/{task_step_id}.csv", index=False, escapechar='\\')

        return f"{self.base_path}/storage/{task_step_id}.csv"