from queue import Queue
from threading import Thread
import traceback
import threading 
from dreamsboard.engine.storage.storage_context import StorageContext
from langchain_community.chat_models import ChatOpenAI
import logging 
import os 
from dreamsboard.engine.task_engine_builder.core import CodeGeneratorBuilder
from dreamsboard.document_loaders.structured_storyboard_loader import LinkedListNode

from dreamsboard.engine.generate.code_generate import QueryProgramGenerator, BaseProgramGenerator
from langchain.schema import AIMessage 
from langchain.prompts import PromptTemplate

from dreamsboard.engine.memory.mctsr.prompt import (
    gpt_prompt_config,
    RefineResponse,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


class Iteratorize:
    """
    这个类会将一个任务封装为延迟执行的迭代器（Generator）。
    任务不会在注册时立即启动线程，而是在 __iter__ 方法第一次被调用时才会启动线程。
    每个任务会检查资源锁，确保同一资源不会被多个任务同时访问。
    """
    def __init__(self, func, resource_id, kwargs={}, resource_lock=None):
        self.mfunc = func
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False
        self.resource_id = resource_id
        self.resource_lock = resource_lock  # 传入具体的锁，而不是整个lock_dict
        self.thread_started = False  # 控制线程是否已经启动

        self._callback = self._callback_func()  # 保存回调函数
        self.thread = None  # 初始化线程，暂时不启动

    def _callback_func(self):
        """回调函数用于收集结果，并控制迭代的结束"""
        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)
        return _callback

    def _run(self):
        """在迭代开始时启动线程并执行任务"""
        # 获取锁，确保同一资源不会被多个任务同时执行
        with self.resource_lock:  # 使用传入的锁来确保资源同步
            try:
                # 执行任务，传入回调函数
                self.mfunc(callback=self._callback, resource_id=self.resource_id, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
            finally:
                # 任务结束，放入sentinel来标识结束
                self.q.put(self.sentinel)

    def __iter__(self):
        """控制迭代器，第一次迭代时启动线程"""
        if not self.thread_started:
            self.thread = Thread(target=self._run)  # 延迟创建线程
            self.thread.start()
            self.thread_started = True

        return self


    def __next__(self):
        """ Fetch the next value from the queue or stop if sentinel is received. """
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        """ Cleanup resources, stop the task if needed. """
        self.stop_now = True

    def __enter__(self):
        """ Used for context management. """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Stop the iteration when exiting context. """
        self.stop_now = True



class EventManager:
    def __init__(self):
        self.lock_dict = {}  # 用来存储资源ID对应的锁
        self.tasks = []
        self.results = []
        self.lock = threading.Lock()  # 添加锁来同步对lock_dict的访问

    def register_event(self, task_func, resource_id, kwargs=None):
        """ 注册事件并绑定资源ID，如果资源ID对应的锁已被占用，等待直到资源释放 """
        kwargs = kwargs or {}
        
        # 使用锁来同步对lock_dict的访问
        with self.lock:
            if resource_id not in self.lock_dict:
                self.lock_dict[resource_id] = threading.Lock()  # 为每个资源ID创建一个锁

        # 创建Iteratorize任务并传入对应的资源锁
        iteratorize_task = Iteratorize(task_func, resource_id, kwargs, self.lock_dict[resource_id])
        self.tasks.append(iteratorize_task)

    def execute_all_events(self):
        """ 执行所有注册的事件并收集结果 """
        for task in self.tasks:
            for result in task:
                self.results.append(result)

    def get_results(self):
        """ 获取所有任务执行的结果 """
        return self.results

        

# Example of how to use the EventManager and Iteratorize

def task_function(callback, resource_id, **kwargs):

    code_gen_builder = CodeGeneratorBuilder.from_template(nodes=[], storage_context=kwargs.get("storage_context"))
    _base_render_data = {
        'system_prompt': kwargs.get("system_prompt"),
        'messages': [kwargs.get("user_prompt")]
    }
    code_gen_builder.add_generator(BaseProgramGenerator.from_config(cfg={
        "code_file": "base_template_system.py-tpl",
        "render_data": _base_render_data,
    }))

    executor = code_gen_builder.build_executor(
        llm_runable=kwargs.get("llm_runable"),
        messages=[]
    )
    executor.execute()
    _ai_message = executor.chat_run()

    logger.info("\033[1;32m" + f"{resource_id}: {_ai_message}" + "\033[0m")
    assert executor._ai_message is not None 

    callback(_ai_message)
    
def main():
    event_manager = EventManager()
    
    llm_runable = ChatOpenAI(
        openai_api_base=os.environ.get("API_BASE"),
        model=os.environ.get("API_MODEL"),
        openai_api_key=os.environ.get("API_KEY"),
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    
 
    evaluate_system_prompt_template = PromptTemplate(
        input_variables=[
            "problem", 
            "answer"
        ],
        template=os.environ.get(
            "evaluate_system_prompt_data", gpt_prompt_config.evaluate_system_prompt_data
        )
    )

    user_prompt = evaluate_system_prompt_template.format(
        problem="任务1",
        answer="结束",
    )
    system_prompt = gpt_prompt_config.evaluate_system_prompt


    storage_context = StorageContext.from_defaults(
        persist_dir=f"/mnt/ceph/develop/jiawei/InterpretationoDreams/src/docs/doubao/60f9b7459a7749597e7efa71d1747bc4/storage/5a31ecd7-6ffe-4497-b4f7-cbbd113d922f"
    )
    # Register tasks for execution
    event_manager.register_event(
        task_function, 
        resource_id="resource_1",          
        kwargs={
            "llm_runable": llm_runable,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "storage_context": storage_context,
    }) 
    
    event_manager.register_event(
        task_function, 
        resource_id="resource_1",          
        kwargs={
            "llm_runable": llm_runable,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "storage_context": storage_context,
    }) 
    event_manager.register_event(
        task_function, 
        resource_id="resource_2",          
        kwargs={
            "llm_runable": llm_runable,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "storage_context": storage_context,
    }) 
    # Execute all tasks
    event_manager.execute_all_events()
    
    # Retrieve and print results
    results = event_manager.get_results()
    print("Final Results:", results)

if __name__ == "__main__":
    main()