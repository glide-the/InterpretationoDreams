# 任务引擎构建器核心类，用于构建任务引擎
# 1、对任务按照提示词要求进行扩写，将扩写任务步骤收集 （src/dreamsboard/dreamsboard/engine/entity/task_step、src/dreamsboard/tests/test_kor/test_kor3.py）

# 2、收集每个任务后存储到磁盘（src/dreamsboard/dreamsboard/engine/storage/task_step_store）

# 3、对每个子任务载入会话场景，然后按照扩写任务步骤构建，MCTS任务

from langchain.chains import LLMChain
from dreamsboard.document_loaders.protocol.ner_protocol import TaskStepNode
from dreamsboard.engine.storage.storage_context import BaseTaskStepStore
from dreamsboard.engine.engine_builder import CodeGeneratorBuilder

class TaskEngineBuilder:
    code_gen_builder: CodeGeneratorBuilder
    task_step_store: BaseTaskStepStore
    task_step_id: str

    def __init__(self,
                 start_task_context: str,
                 task_step_store: BaseTaskStepStore,
                 task_step_id: str
    ):
        """执行资源
        AEMORepresentationChain 
        task_step_store[task_step_id]

        根据任务步骤，构建场景加载模块，生成资源文件csv
        根据每个任务，载入StructuredDreamsStoryboard 会话场景
        按照扩写任务步骤构建MCTS任务
        
        """
        self.start_task_context = start_task_context
        self.task_step_store = task_step_store
        self.task_step_id = task_step_id

   
    def build_task_engine(self, task_step_node: TaskStepNode):
        pass
