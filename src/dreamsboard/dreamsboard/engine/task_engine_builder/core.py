# 任务引擎构建器核心类，用于构建任务引擎
 
# 3、对每个子任务载入会话场景，然后按照扩写任务步骤构建，MCTS任务

from langchain.schema.language_model import BaseLanguageModel
from dreamsboard.engine.storage.task_step_store.types import BaseTaskStepStore
from dreamsboard.engine.engine_builder import CodeGeneratorBuilder

from dreamsboard.dreams.task_step_to_question_chain.base import TaskStepToQuestionChain
from dreamsboard.dreams.task_step_to_question_chain.weaviate.context_collections import init_context_connect

from dreamsboard.document_loaders import StructuredStoryboardCSVBuilder
from dreamsboard.dreams.builder_cosplay_code.base import StructuredDreamsStoryboard
from dreamsboard.dreams.dreams_personality_chain.base import StoryBoardDreamsGenerationChain
from dreamsboard.document_loaders.structured_storyboard_loader import StructuredStoryboard

from dreamsboard.engine.entity.dreams_personality.dreams_personality import DreamsPersonalityNode
from dreamsboard.engine.generate.code_generate import QueryProgramGenerator, EngineProgramGenerator
from dreamsboard.engine.loading import load_store_from_storage
from dreamsboard.engine.storage.dreams_analysis_store.simple_dreams_analysis_store import SimpleDreamsAnalysisStore
from dreamsboard.engine.storage.storage_context import StorageContext
from dreamsboard.engine.utils import concat_dirs
import logging 
from dreamsboard.engine.memory.mctsr.mctsr import MCTSNode, MCTSrStoryboard
from langchain.schema import AIMessage
from dreamsboard.engine.storage.task_step_store.types import DEFAULT_PERSIST_FNAME

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


class TaskEngineBuilder:

    """执行会话场景资源初始化，构建MCTS任务
    AEMORepresentationChain 
    task_step_store[task_step_id]

    根据任务步骤，构建场景加载模块，生成资源文件csv
    根据每个任务，载入StructuredDreamsStoryboard 会话场景
    按照扩写任务步骤构建MCTS任务
    
    """
    cross_encoder_path: str
    storage_context: StorageContext
    task_step_store: BaseTaskStepStore
    task_step_id: str
    csv_file_path: str
    dreams_analysis_store: SimpleDreamsAnalysisStore
    storyboard_executor: StructuredDreamsStoryboard
    engine_template_render_data: dict = {}

    def __init__(self,
                llm: BaseLanguageModel,
                cross_encoder_path: str,
                start_task_context: str,
                task_step_store: BaseTaskStepStore,
                task_step_id: str,
                engine_template_render_data: dict = {}
    ):
        self.start_task_context = start_task_context
        self.task_step_store = task_step_store
        self.task_step_id = task_step_id
        self.llm = llm
        self.engine_template_render_data = engine_template_render_data
        self.cross_encoder_path = cross_encoder_path
        self.client = None
        self.task_step_to_question_chain = None
        self.csv_file_path = None
        self.storyboard_executor = None
        self.storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/{self.task_step_id}"
        )
        self.storage_context.task_step_store = self.task_step_store

    def check_engine_init(self):
        """
        检查引擎是否初始化
        """
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=f"./storage/{self.task_step_id}"
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
			1、对这个提示词所要求的输入拆分成子任务， 
			2、对每个子任务指令转换为子问题，召回问题前3条，
		》StoryBoardDreamsGenerationChain
			对每个子任务载入会话场景
 
        """
              
        client = init_context_connect()


        self.task_step_to_question_chain = TaskStepToQuestionChain.from_task_step_to_question_chain(
            llm=self.llm, 
            task_step_store=self.task_step_store,
            client=client,
            cross_encoder_path=self.cross_encoder_path
        )
        self.task_step_to_question_chain.invoke_task_step_to_question(self.task_step_id)
        self.task_step_to_question_chain.invoke_task_step_question_context(self.task_step_id)
        self.csv_file_path = self.task_step_to_question_chain.export_csv_file_path(self.task_step_id)
   
 
    def init_task_engine_dreams(self, allow_init: bool = True) -> None:
        """
        初始化场景加载资源
        如果allow_init为True，则清空存储，重新初始化
        """ 
        if self.csv_file_path is None:
            raise ValueError("csv_file_path is None, please invoke init_task_engine first")
        
        dreams_analysis_store = SimpleDreamsAnalysisStore.from_persist_dir(persist_dir=f"./storage/{self.task_step_id}")
        if allow_init:
            analysis_ids = list(dreams_analysis_store.analysis_all.keys())
            for analysis_id in analysis_ids:
                dreams_analysis_store.delete_analysis(analysis_id=analysis_id)

            dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
                llm=self.llm, csv_file_path=self.csv_file_path
            )

            output = dreams_generation_chain.run() 
            # 拼接dreams_guidance_context和dreams_personality_context两个字典
            dreams_generation = {}
            dreams_generation.update(output.get("dreams_guidance_context"))
            dreams_generation.update(output.get("dreams_personality_context"))
            dreams = DreamsPersonalityNode.from_config(cfg=dreams_generation)
            dreams_analysis_store.add_analysis([dreams])
            dreams_analysis_store_path = concat_dirs(dirname=f"./storage/{self.task_step_id}", basename="dreams_analysis_store.json")
            dreams_analysis_store.persist(persist_path=dreams_analysis_store_path)

        self.storage_context.dreams_analysis_store = dreams_analysis_store
        self.storage_context.persist(persist_dir=f"./storage/{self.task_step_id}")
            
    def init_task_engine_storyboard_executor(self) -> None:
        """
        构建会话场景执行器
        """
        
        if self.csv_file_path is None:
            raise ValueError("csv_file_path is None, please invoke init_task_engine first")
        
        if self.storage_context is None or self.storage_context.dreams_analysis_store is None:
            raise ValueError("storage_context is None or dreams_analysis_store is None, please invoke init_task_engine_dreams first")
        
        for val in self.storage_context.dreams_analysis_store.analysis_all.values():
            dreams_guidance_context = val.dreams_guidance_context
            dreams_personality_context = val.dreams_personality_context

        csv_builder = StructuredStoryboardCSVBuilder.form_builder(csv_file_path=self.csv_file_path)
        csv_builder.load()
        self.storyboard_executor = StructuredDreamsStoryboard.form_builder(
            llm=self.llm,
            builder=csv_builder,
            dreams_guidance_context=dreams_guidance_context,
            dreams_personality_context=dreams_personality_context,
            guidance_llm=self.llm,
            personality_llm=self.llm,
            user_id=self.task_step_id
        )

    def storyboard_code_gen_builder(self) -> CodeGeneratorBuilder:
        """
        构建会话场景执行器
        """
        if self.storage_context is None:
            raise ValueError("storage_context is None, please invoke init_task_engine first")
        
        try:
            code_gen_builder = load_store_from_storage(storage_context=self.storage_context)
        except:
            code_gen_builder = self.storyboard_executor.loader_cosplay_builder(
                storage_context=self.storage_context, 
                engine_template_render_data=self.engine_template_render_data
            )
            
        code_gen_builder.storage_context.dreams_analysis_store = self.storage_context.dreams_analysis_store
        code_gen_builder.storage_context.template_store = self.storage_context.template_store
        code_gen_builder.storage_context.index_store = self.storage_context.index_store
        code_gen_builder.storage_context.task_step_store = self.storage_context.task_step_store
        code_gen_builder.storage_context.persist(persist_dir=f"./storage/{self.task_step_id}")

        return code_gen_builder
    
    def generate_step_answer(self, code_gen_builder: CodeGeneratorBuilder) -> str:
        """
        生成当前任务的答案
        """
        task_step = self.task_step_store.get_task_step(self.task_step_id)
        ai_message = self._get_ai_message(
            user_prompt=task_step.task_step_question,
            code_gen_builder=code_gen_builder, 
            engine_template_render_data=self.engine_template_render_data, 
        )
        logger.info(ai_message.content)
        task_step.task_step_question_answer = ai_message.content
        self.task_step_store.add_task_step([task_step])
        # 每处理一个任务步骤，就持久化一次
        task_step_store_path = concat_dirs(dirname=f"./storage/{self.task_step_id}", basename=DEFAULT_PERSIST_FNAME)
        self.task_step_store.persist(persist_path=task_step_store_path) 
        return ai_message.content
    
    def _get_ai_message(self, 
                        user_prompt: str,
                        code_gen_builder: CodeGeneratorBuilder,
                        engine_template_render_data: dict
        ) -> AIMessage:
        """
        获取AI消息
        """
                 
        code_gen_builder.add_generator(QueryProgramGenerator.from_config(cfg={
            "query_code_file": "query_template.py-tpl",
            "render_data": {
                'cosplay_role': 'user',
                'message': user_prompt
            },
        }))

        code_gen_builder.add_generator(EngineProgramGenerator.from_config(cfg={
            "engine_code_file": "simple_engine_template.py-tpl",
            "render_data": engine_template_render_data
        }))

        executor = code_gen_builder.build_executor()
        logger.info(executor.executor_code)

        executor.execute()
        _ai_message = executor.chat_run()

        logger.info(executor._ai_message)
        assert executor._ai_message is not None
        
        code_gen_builder.remove_last_generator()
        code_gen_builder.remove_last_generator()

        return _ai_message


    def get_mcts_node(self, code_gen_builder: CodeGeneratorBuilder) -> MCTSrStoryboard:
        """
        构建MCTS树, 初始化当前任务相关的MCTS节点，并返回MCTS执行器
        """
        task_step_all = self.task_step_store.task_step_all
        task_step_all_list = [val.__dict__ for val in list(task_step_all.values())]
        structured_storyboard = StructuredStoryboard(json_data=task_step_all_list) 
        linked_list_node = structured_storyboard.get_task_step_node(self.task_step_id)
        
        mcts_node = MCTSNode(
            answer=linked_list_node.task_step_question_answer,
            linked_list_node=linked_list_node,
            code_gen_builder=code_gen_builder, 
            engine_template_render_data=self.engine_template_render_data,
            parent=None, 
            children=[], 
            visits=0, 
            Q=0, 
            reward_samples=[]
        )
        # 构建MCTS树, 初始化当前节点的顶层节点
        while linked_list_node is not None and linked_list_node.prev is not None:
            linked_list_node = linked_list_node.prev
            parent_node = MCTSNode(
                answer=linked_list_node.task_step_question_answer, 
                linked_list_node=linked_list_node,
                code_gen_builder=code_gen_builder, 
                engine_template_render_data=self.engine_template_render_data,
                parent=None, 
                children=[], 
                visits=0, 
                Q=0, 
                reward_samples=[]
            )
            mcts_node.parent = parent_node

            
        task_step = self.task_step_store.get_task_step(self.task_step_id)
        mctsr = MCTSrStoryboard(
            problem=task_step.task_step_name, 
            max_rollouts=10
        )
        mctsr.initialize(mcts_node)
        mctsr.print()
        return mctsr
