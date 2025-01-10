from __future__ import annotations
from typing import List, Sequence
 
from dreamsboard.engine.task_engine_builder.core import TaskEngineBuilder
 
from dreamsboard.document_loaders import KorLoader
from dreamsboard.dreams.aemo_representation_chain.base import AEMORepresentationChain
from langchain.chains import LLMChain
from langchain.schema.language_model import BaseLanguageModel
from dreamsboard.engine.entity.task_step.task_step import TaskStepNode
from dreamsboard.engine.storage.task_step_store.types import BaseTaskStepStore
import queue
import logging

from dreamsboard.engine.loading import load_store_from_storage
from dreamsboard.engine.storage.storage_context import StorageContext
from dreamsboard.engine.storage.task_step_store.simple_task_step_store import SimpleTaskStepStore
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


class StructuredTaskStepStoryboard:
    """
    对任务进行规划，生成段落之间组成一个动态上下文
    任务二 需求：

        11、对任务按照提示词要求进行扩写，将扩写任务步骤收集 （src/dreamsboard/dreamsboard/engine/entity/task_step、src/dreamsboard/tests/test_kor/test_kor3.py）

        2、收集每个任务后存储到磁盘（src/dreamsboard/dreamsboard/engine/storage/task_step_store）

        3、对每个子任务载入会话场景，然后按照扩写任务步骤构建，MCTS任务
        导出代码
    """ 
    task_step_store: BaseTaskStepStore
    start_task_context: str
    cross_encoder_path: str
    llm: BaseLanguageModel
    aemo_representation_chain: AEMORepresentationChain

    def __init__(self,
                 llm: BaseLanguageModel,
                 cross_encoder_path: str,
                 start_task_context: str,
                 aemo_representation_chain: AEMORepresentationChain,
                 task_step_store: BaseTaskStepStore,
                 ):
        """

        :param start_task_context: 开始任务
        :param aemo_representation_chain: 情感表征链
        """
        self.llm = llm
        self.cross_encoder_path = cross_encoder_path
        self.start_task_context = start_task_context
        self.aemo_representation_chain = aemo_representation_chain
        self.task_step_store = task_step_store

    @classmethod
    def form_builder(cls,
                     llm: BaseLanguageModel,
                     cross_encoder_path: str,
                     start_task_context: str, 
                     kor_dreams_task_step_llm: BaseLanguageModel = None,
                     task_step_store: BaseTaskStepStore = None,
                     ) -> StructuredTaskStepStoryboard: 
        aemo_representation_chain = AEMORepresentationChain.from_aemo_representation_chain(
            llm=llm,
            start_task_context=start_task_context,
            kor_dreams_task_step_llm=kor_dreams_task_step_llm
        )
        return cls(llm=llm,
                   cross_encoder_path=cross_encoder_path,
                   start_task_context=start_task_context,
                   aemo_representation_chain=aemo_representation_chain,
                   task_step_store=task_step_store)


    def loader_task_step_iter_builder(self, engine_template_render_data: dict = {}, allow_init: bool = True) -> queue.Queue[TaskEngineBuilder]:
        """
        加载任务步骤迭代器
        :param allow_init: 是否初始化
        """
        iter_builder_queue = queue.Queue()
        if not allow_init: 
            for task_step_id, task_step in self.task_step_store.task_step_all.items():
                task_step_id = task_step.node_id
                
                task_step_store = SimpleTaskStepStore.from_persist_dir(f"./storage/{task_step_id}")
                iter_builder_queue.put(TaskEngineBuilder(
                    llm=self.llm,
                    cross_encoder_path=self.cross_encoder_path,
                    start_task_context=self.start_task_context,
                    task_step_store=task_step_store,
                    task_step_id=task_step_id,
                    engine_template_render_data=engine_template_render_data
                ))

        else:
            result = self.aemo_representation_chain.invoke_aemo_representation_context()
            task_step_iter = self.aemo_representation_chain.invoke_kor_dreams_task_step_context(
                aemo_representation_context=result.get('aemo_representation_context')
            )
            for task_step in task_step_iter:
                    
                task_step = TaskStepNode.from_config(cfg={
                    "start_task_context": self.start_task_context,
                    "aemo_representation_context": result.get('aemo_representation_context'),
                    "task_step_name": task_step.task_step_name,
                    "task_step_description": task_step.task_step_description,
                    "task_step_level": task_step.task_step_level
                })
                task_step_id = task_step.node_id 
                iter_builder_queue.put(TaskEngineBuilder(
                    llm=self.llm,
                    cross_encoder_path=self.cross_encoder_path,
                    start_task_context=self.start_task_context,
                    task_step_store=self.task_step_store,
                    task_step_id=task_step_id,
                    engine_template_render_data=engine_template_render_data
                ))

        return iter_builder_queue
