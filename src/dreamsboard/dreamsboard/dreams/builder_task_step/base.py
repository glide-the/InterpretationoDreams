from __future__ import annotations

import logging
import os
import queue
import re
import torch
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from sentence_transformers import CrossEncoder

from dreamsboard.document_loaders import KorLoader
from dreamsboard.dreams.aemo_representation_chain.base import AEMORepresentationChain
from dreamsboard.dreams.task_step_to_question_chain.weaviate.prepare_load import (
    get_query_hash,
)
from dreamsboard.engine.entity.task_step.task_step import TaskStepNode
from dreamsboard.engine.storage.task_step_store.simple_task_step_store import (
    SimpleTaskStepStore,
)
from dreamsboard.engine.storage.task_step_store.types import (
    DEFAULT_PERSIST_FNAME,
    BaseTaskStepStore,
)
from dreamsboard.engine.task_engine_builder.core import TaskEngineBuilder
from dreamsboard.engine.utils import concat_dirs
from dreamsboard.vector.base import CollectionService
from dreamsboard.vector.faiss_kb_service import FaissCollectionService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


class StructuredTaskStepStoryboard:
    """
    对任务进行规划，生成段落之间组成一个动态上下文
    任务：

        1、对任务按照提示词要求进行扩写，将扩写任务步骤收集 （src/dreamsboard/dreamsboard/engine/entity/task_step、src/dreamsboard/tests/test_kor/test_kor3.py）

        2、收集每个任务后存储到磁盘（src/dreamsboard/dreamsboard/engine/storage/task_step_store）

        3、对每个子任务载入会话场景，然后按照扩写任务步骤构建，MCTS任务 loader_task_step_iter_builder

    """

    base_path: str
    task_step_store: BaseTaskStepStore
    start_task_context: str
    cross_encoder: CrossEncoder
    collection: CollectionService
    data_base: str
    llm_runable: Runnable[LanguageModelInput, BaseMessage]
    kor_dreams_task_step_llm: Runnable[LanguageModelInput, BaseMessage]
    aemo_representation_chain: AEMORepresentationChain

    def __init__(
        self,
        base_path: str,
        llm_runable: Runnable[LanguageModelInput, BaseMessage],
        start_task_context: str,
        cross_encoder: CrossEncoder,
        collection: CollectionService,
        aemo_representation_chain: AEMORepresentationChain,
        task_step_store: BaseTaskStepStore,
        data_base: str,
        kor_dreams_task_step_llm: Runnable[LanguageModelInput, BaseMessage]= None,
    ):
        """

        :param base_path: 基础路径
        :param start_task_context: 开始任务
        :param aemo_representation_chain: 情感表征链
        """
        self.base_path = base_path
        self.llm_runable = llm_runable
        self.cross_encoder = cross_encoder
        self.collection = collection
        self.start_task_context = start_task_context
        self.aemo_representation_chain = aemo_representation_chain
        self.task_step_store = task_step_store
        self.data_base = data_base
        self.kor_dreams_task_step_llm = kor_dreams_task_step_llm

    @classmethod
    def form_builder(
        cls,
        llm_runable: Runnable[LanguageModelInput, BaseMessage],
        cross_encoder_path: str,
        embed_model_path: str,
        start_task_context: str,
        data_base: str = "search_papers",
        kor_dreams_task_step_llm: Runnable[LanguageModelInput, BaseMessage]
        | None = None,
        task_step_store: BaseTaskStepStore | None = None,
    ) -> StructuredTaskStepStoryboard:
        aemo_representation_chain = (
            AEMORepresentationChain.from_aemo_representation_chain(
                llm_runable=llm_runable,
                start_task_context=start_task_context,
                kor_dreams_task_step_llm=kor_dreams_task_step_llm,
            )
        )

        base_path = f"./{get_query_hash(start_task_context)}/"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if task_step_store is None:
            task_step_store = SimpleTaskStepStore.from_persist_dir(
                f"./{base_path}/storage"
            )

        collection_id = get_query_hash(start_task_context)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        collection = FaissCollectionService(
            kb_name=collection_id,
            embed_model=embed_model_path,
            vector_name="samples",
            device=device,
        )
        cross_encoder = CrossEncoder(
            cross_encoder_path,
            automodel_args={"torch_dtype": "auto"},
            trust_remote_code=True,
        )

        return cls(
            base_path=base_path,
            llm_runable=llm_runable,
            kor_dreams_task_step_llm=kor_dreams_task_step_llm,
            start_task_context=start_task_context,
            cross_encoder=cross_encoder,
            collection=collection,
            data_base=data_base,
            aemo_representation_chain=aemo_representation_chain,
            task_step_store=task_step_store,
        )

    def loader_task_step_iter_builder(
        self, allow_init: bool = True
    ) -> queue.Queue[TaskEngineBuilder]:
        """
        加载任务步骤迭代器
        :param allow_init: 是否初始化
        """
        iter_builder_queue = queue.Queue()
        if not allow_init:
            for task_step_id, task_step in self.task_step_store.task_step_all.items():
                task_step_id = task_step.node_id

                # 从存储中获取详细的任务,设置到全局上下文
                task_step_store_node = SimpleTaskStepStore.from_persist_dir(
                    f"{self.base_path}/storage/{task_step_id}"
                )
                task_step_node = task_step_store_node.get_task_step(task_step_id)

                task_step_store_node.add_task_step([task_step_node])
                task_step_store_path = concat_dirs(
                    dirname=f"{self.base_path}/storage/{task_step_id}",
                    basename=DEFAULT_PERSIST_FNAME,
                )

                self.task_step_store.persist(persist_path=task_step_store_path)
                task_step_store_path = concat_dirs(
                    dirname=f"{self.base_path}/storage", basename=DEFAULT_PERSIST_FNAME
                )
                self.task_step_store.persist(persist_path=task_step_store_path)
                iter_builder_queue.put(
                    TaskEngineBuilder(
                        llm_runable=self.llm_runable,
                        kor_dreams_task_step_llm=self.kor_dreams_task_step_llm,
                        cross_encoder=self.cross_encoder,
                        collection=self.collection,
                        start_task_context=self.start_task_context,
                        task_step_store=self.task_step_store,
                        task_step_id=task_step_id,
                        base_path=self.base_path,
                        data_base=self.data_base,
                    )
                )

        else:
            result = self.aemo_representation_chain.invoke_aemo_representation_context()
    
            cleaned_text = re.sub(r'◁think▷.*?◁/think▷', '', result["aemo_representation_context"], flags=re.DOTALL)
            task_step_iter = (
                self.aemo_representation_chain.invoke_kor_dreams_task_step_context(
                    aemo_representation_context=cleaned_text
                )
            )
            for task_step in task_step_iter:
                task_step_node = TaskStepNode.from_config(
                    cfg={
                        "start_task_context": self.start_task_context,
                        "aemo_representation_context": result.get(
                            "aemo_representation_context"
                        ),
                        "task_step_name": task_step.task_step_name,
                        "task_step_description": task_step.task_step_description,
                        "task_step_level": task_step.task_step_level,
                    }
                )
                task_step_id = task_step_node.node_id

                self.task_step_store.add_task_step([task_step_node])

                task_step_store_path = concat_dirs(
                    dirname=f"{self.base_path}/storage/{task_step_id}",
                    basename=DEFAULT_PERSIST_FNAME,
                )

                self.task_step_store.persist(persist_path=task_step_store_path)
                task_step_store_path = concat_dirs(
                    dirname=f"{self.base_path}/storage", basename=DEFAULT_PERSIST_FNAME
                )
                self.task_step_store.persist(persist_path=task_step_store_path)
                iter_builder_queue.put(
                    TaskEngineBuilder(
                        llm_runable=self.llm_runable,
                        kor_dreams_task_step_llm=self.kor_dreams_task_step_llm,
                        cross_encoder=self.cross_encoder,
                        collection=self.collection,
                        start_task_context=self.start_task_context,
                        task_step_store=self.task_step_store,
                        task_step_id=task_step_id,
                        base_path=self.base_path,
                        data_base=self.data_base,
                    )
                )

        return iter_builder_queue
