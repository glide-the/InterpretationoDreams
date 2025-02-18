# 任务引擎构建器核心类，用于构建任务引擎

# 3、对每个子任务载入会话场景，然后按照扩写任务步骤构建，MCTS任务

import logging
import threading
import re

from langchain.schema import AIMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.runnables import Runnable
from sentence_transformers import CrossEncoder

from dreamsboard.common.callback import call_func
from dreamsboard.document_loaders import StructuredStoryboardCSVBuilder
from dreamsboard.dreams.builder_cosplay_code.base import StructuredDreamsStoryboard
from dreamsboard.dreams.dreams_personality_chain.base import (
    StoryBoardDreamsGenerationChain,
)
from dreamsboard.dreams.task_step_to_question_chain.base import TaskStepToQuestionChain
from dreamsboard.engine.engine_builder import CodeGeneratorBuilder
from dreamsboard.engine.entity.dreams_personality.dreams_personality import (
    DreamsPersonalityNode,
)
from dreamsboard.engine.generate.code_generate import (
    EngineProgramGenerator,
    QueryProgramGenerator,
)
from dreamsboard.engine.loading import load_store_from_storage
from dreamsboard.engine.memory.mctsr.mctsr import MCTSNode, MCTSrStoryboard
from dreamsboard.engine.storage.dreams_analysis_store.simple_dreams_analysis_store import (
    SimpleDreamsAnalysisStore,
)
from dreamsboard.engine.storage.storage_context import StorageContext
from dreamsboard.engine.storage.task_step_store.types import (
    DEFAULT_PERSIST_FNAME,
    BaseTaskStepStore,
)
from dreamsboard.engine.utils import concat_dirs
from dreamsboard.vector.base import CollectionService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


class TaskEngineBuilder:

    """TaskEngineBuilder 场景加载模块
                执行会话场景资源初始化，构建MCTS任务

    根据任务步骤，构建场景加载模块，生成资源文件csv
    根据每个任务，载入StructuredDreamsStoryboard 会话场景
    按照扩写任务步骤构建MCTS任务

                输入：
                        task_step_id
                        task_step_store: 任务存储器（SimpleTaskStepStore）
                        start_task_context： 起始任务
                        llm： 模型


    """

    base_path: str
    cross_encoder: CrossEncoder
    data_base: str
    collection: CollectionService
    storage_context: StorageContext
    task_step_store: BaseTaskStepStore
    task_step_id: str
    csv_file_path: str
    storyboard_executor: StructuredDreamsStoryboard
    _llm_runable: Runnable[LanguageModelInput, BaseMessage]
    _kor_dreams_task_step_llm: Runnable[LanguageModelInput, BaseMessage]

    def __init__(
        self,
        base_path: str,
        llm_runable: Runnable[LanguageModelInput, BaseMessage],
        cross_encoder: CrossEncoder,
        collection: CollectionService,
        start_task_context: str,
        task_step_store: BaseTaskStepStore,
        task_step_id: str,
        data_base: str = "search_papers",
        kor_dreams_task_step_llm: Runnable[LanguageModelInput, BaseMessage]= None,
    ):
        self.base_path = base_path
        self.start_task_context = start_task_context
        self.task_step_store = task_step_store
        self.task_step_id = task_step_id
        self.data_base = data_base
        self._llm_runable = llm_runable
        self.cross_encoder = cross_encoder
        self.collection = collection
        self.client = None
        self.task_step_to_question_chain = None
        self.csv_file_path = None
        self.storyboard_executor = None
        self.storage_context = StorageContext.from_defaults(
            persist_dir=f"{self.base_path}/storage/{self.task_step_id}"
        )
        self.storage_context.task_step_store = self.task_step_store
        self._kor_dreams_task_step_llm = kor_dreams_task_step_llm

    @property
    def llm_runable(self) -> Runnable[LanguageModelInput, BaseMessage]:
        return self._llm_runable

    @llm_runable.setter
    def llm_runable(
        self, llm_runable: Runnable[LanguageModelInput, BaseMessage]
    ) -> None:
        self._llm_runable = llm_runable
    @property
    def kor_dreams_task_step_llm(self) -> Runnable[LanguageModelInput, BaseMessage]:
        return self._kor_dreams_task_step_llm

    @kor_dreams_task_step_llm.setter
    def kor_dreams_task_step_llm(
        self, kor_dreams_task_step_llm: Runnable[LanguageModelInput, BaseMessage]
    ) -> None:
        self._kor_dreams_task_step_llm = kor_dreams_task_step_llm

    def check_engine_init(self):
        """
        检查引擎是否初始化
        """
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{self.base_path}/storage/{self.task_step_id}"
            )
            if storage_context.task_step_store is None:
                return False
            if storage_context.dreams_analysis_store is None:
                return False
            if storage_context.template_store is None:
                return False
            if storage_context.index_store is None:
                return False

            code_gen_builder = load_store_from_storage(storage_context=storage_context)
            if code_gen_builder is None:
                return False
            else:
                return True
        except:
            return False

    def init_task_engine(self):
        """

        初始化任务引擎
        》TaskStepToQuestionChain
                输入：
                        client： 矢量库客户端
                        llm： 模型
                                        invoke_task_step_to_question：1、 对开始任务进行抽取，得到任务步骤，提示词所要求的输入拆分成子任务，
                                        invoke_task_step_question_context： 2、对每个子任务指令转换为子问题，召回问题前3条，对任务步骤进行抽取，得到任务步骤的上下文
                                        export_csv_file_path: 3、对召回内容与问题 导出csv文件

        """

        self.task_step_to_question_chain = (
            TaskStepToQuestionChain.from_task_step_to_question_chain(
                base_path=self.base_path,
                start_task_context=self.start_task_context,
                llm_runable=self.llm_runable,
                task_step_store=self.task_step_store,
                collection=self.collection,
                cross_encoder=self.cross_encoder,
                data_base=self.data_base,
            )
        )
        self.task_step_to_question_chain.invoke_task_step_to_question(self.task_step_id)
        self.task_step_to_question_chain.invoke_task_step_question_context(
            self.task_step_id
        )
        self.csv_file_path = self.task_step_to_question_chain.export_csv_file_path(
            self.task_step_id
        )

    def init_task_engine_dreams(self, allow_init: bool = True) -> None:
        """
        初始化场景加载资源StoryBoardDreamsGenerationChain
        如果allow_init为True，则清空存储，重新初始化

        对每个子任务通过职业提示词，载入会话场景
            1、构建场景信息（story_scenario_context），提示词（STORY_BOARD_SCENE_TEMPLATE）
            2、对任务上下文(story_board_summary_context)，构建第一人称数据(scene_monologue_context),提示词（STORY_BOARD_SUMMARY_CONTEXT_TEMPLATE）
            3、对任务上下文(story_board_summary_context)，获取任务分析(evolutionary_step), 提示词（EDREAMS_EVOLUTIONARY_TEMPLATE）
            4、对任务分析(evolutionary_step)，分析对话预设信息（性格）， 提示词（EDREAMS_PERSONALITY_TEMPLATE）
            5、对任务上下文(story_board_summary_context)，场景信息story_scenario_context, 第一人称数据(scene_monologue_context)，
            生成关于人物职业的引导话术，提示词（DREAMS_GEN_TEMPLATE）
        """
        if self.csv_file_path is None:
            raise ValueError(
                "csv_file_path is None, please invoke init_task_engine first"
            )

        dreams_analysis_store = SimpleDreamsAnalysisStore.from_persist_dir(
            persist_dir=f"{self.base_path}/storage/{self.task_step_id}"
        )
        if allow_init:
            analysis_ids = list(dreams_analysis_store.analysis_all.keys())
            for analysis_id in analysis_ids:
                dreams_analysis_store.delete_analysis(analysis_id=analysis_id)

            dreams_generation_chain = (
                StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
                    llm_runable=self.llm_runable, csv_file_path=self.csv_file_path
                )
            )

            output = dreams_generation_chain.run()
            # 拼接dreams_guidance_context和dreams_personality_context两个字典
            dreams_generation = {}
            dreams_generation.update(output.get("dreams_guidance_context"))
            dreams_generation.update(output.get("dreams_personality_context"))
            dreams = DreamsPersonalityNode.from_config(cfg=dreams_generation)
            dreams_analysis_store.add_analysis([dreams])
            dreams_analysis_store_path = concat_dirs(
                dirname=f"{self.base_path}/storage/{self.task_step_id}",
                basename="dreams_analysis_store.json",
            )
            dreams_analysis_store.persist(persist_path=dreams_analysis_store_path)

        self.storage_context.dreams_analysis_store = dreams_analysis_store
        self.storage_context.persist(
            persist_dir=f"{self.base_path}/storage/{self.task_step_id}"
        )

    def init_task_engine_storyboard_executor(self) -> None:
        """
        构建会话场景执行器
        """

        if self.csv_file_path is None:
            raise ValueError(
                "csv_file_path is None, please invoke init_task_engine first"
            )

        if (
            self.storage_context is None
            or self.storage_context.dreams_analysis_store is None
        ):
            raise ValueError(
                "storage_context is None or dreams_analysis_store is None, please invoke init_task_engine_dreams first"
            )

        for val in self.storage_context.dreams_analysis_store.analysis_all.values():
            dreams_guidance_context = val.dreams_guidance_context
            dreams_personality_context = val.dreams_personality_context

        csv_builder = StructuredStoryboardCSVBuilder.form_builder(
            csv_file_path=self.csv_file_path
        )
        csv_builder.load()
        self.storyboard_executor = StructuredDreamsStoryboard.form_builder(
            llm_runable=self.llm_runable,
            builder=csv_builder,
            dreams_guidance_context=dreams_guidance_context,
            dreams_personality_context=dreams_personality_context,
            guidance_llm=self.llm_runable if self._kor_dreams_task_step_llm is None else self._kor_dreams_task_step_llm,
            personality_llm=self.llm_runable if self._kor_dreams_task_step_llm is None else self._kor_dreams_task_step_llm,
            user_id=self.task_step_id,
        )

    def storyboard_code_gen_builder(self) -> CodeGeneratorBuilder:
        """
        构建会话场景执行器
        """
        if self.storage_context is None:
            raise ValueError(
                "storage_context is None, please invoke init_task_engine first"
            )

        try:
            code_gen_builder = load_store_from_storage(
                storage_context=self.storage_context
            )
        except:
            code_gen_builder = self.storyboard_executor.loader_cosplay_builder(
                dreams_cosplay_role=self.task_step_id,
                storage_context=self.storage_context,
            )

        code_gen_builder.storage_context.dreams_analysis_store = (
            self.storage_context.dreams_analysis_store
        )
        code_gen_builder.storage_context.template_store = (
            self.storage_context.template_store
        )
        code_gen_builder.storage_context.index_store = self.storage_context.index_store
        code_gen_builder.storage_context.task_step_store = (
            self.storage_context.task_step_store
        )
        code_gen_builder.storage_context.persist(
            persist_dir=f"{self.base_path}/storage/{self.task_step_id}"
        )

        return code_gen_builder

    def generate_step_answer(self, code_gen_builder: CodeGeneratorBuilder) -> str:
        """
        生成当前任务的答案
        """
        task_step = self.task_step_store.get_task_step(self.task_step_id)

        owner = f"register_event thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}")

        results = call_func(
            self._get_ai_message,
            resource_id=f"resource_critic_{self.task_step_id}",
            kwargs={
                "llm_runable": self.llm_runable,
                "code_gen_builder": code_gen_builder,
                "user_prompt": task_step.task_step_question,
            },
        )

        _ai_message = results[0]
        cleaned_text = re.sub(r'◁think▷.*?◁/think▷', '',_ai_message.content, flags=re.DOTALL)
        task_step.task_step_question_answer = cleaned_text
        self.task_step_store.add_task_step([task_step])
        # 每处理一个任务步骤，就持久化一次
        task_step_store_path = concat_dirs(
            dirname=f"{self.base_path}/storage/{self.task_step_id}",
            basename=DEFAULT_PERSIST_FNAME,
        )
        self.task_step_store.persist(persist_path=task_step_store_path)
        return _ai_message.content

    @staticmethod
    def _get_ai_message(callback, resource_id, **kwargs):
        code_gen_builder = kwargs.get("code_gen_builder")

        code_gen_builder.add_generator(
            QueryProgramGenerator.from_config(
                cfg={
                    "query_code_file": "query_template.py-tpl",
                    "render_data": {
                        "cosplay_role": "user",
                        "message": kwargs.get("user_prompt"),
                    },
                }
            )
        )
        executor = code_gen_builder.build_executor(
            llm_runable=kwargs.get("llm_runable"), messages=[]
        )

        executor.execute()
        _ai_message = executor.chat_run()

        assert executor._ai_message is not None

        code_gen_builder.remove_last_generator()

        logger.info("\033[1;32m" + f"{resource_id}: {_ai_message}" + "\033[0m")

        callback(_ai_message)

    def get_mcts_node(self) -> MCTSrStoryboard:
        """
        构建MCTS树, 初始化当前任务相关的MCTS节点，并返回MCTS执行器
        """
        task_step = self.task_step_store.get_task_step(self.task_step_id)
        mcts_node = MCTSNode(
            task_step_id=self.task_step_id,
            answer=task_step.task_step_question_answer,
            parent=None,
            children=[],
            visits=0,
            Q=0,
            reward_samples=[],
        )
        mctsr = MCTSrStoryboard.model_construct(
            base_path=self.base_path,
            task_step_id=self.task_step_id,
            storage_context=self.storage_context,
            root=mcts_node,
            llm_runable=self.llm_runable,
            kor_dreams_task_step_llm=self.kor_dreams_task_step_llm,
            problem=task_step.task_step_name,
            max_rollouts=2,
        )
        return mctsr
