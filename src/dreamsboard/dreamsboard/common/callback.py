from queue import Queue
from threading import Thread
import traceback
import threading 
from dreamsboard.engine.storage.storage_context import StorageContext
from langchain_community.chat_models import ChatOpenAI
import logging 
import os 
import time
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
        """在迭代开始时启动线程并执行任务
        DO：在 EventManager 中，多个事件（task_function）可能会同时尝试获取相同资源的锁（resource_lock），
        但如果一个线程持有某个锁且另一个线程正在等待同一个锁，它们就会产生死锁。
        如果多个任务被注册到相同资源，并且这些任务还相互依赖（或者访问同一个资源），则可能会形成死锁链。
        解决方案: 对资源锁增加一个超时机制，若在一定时间内无法获得锁，可以进行回滚，避免无限期阻塞。
        """
        # 获取锁，确保同一资源不会被多个任务同时执行
        lock_acquired = False
        start_time = time.time()
        timeout = 5  # 设置一个超时限制

        # 尝试获取锁，避免死锁
        while not lock_acquired:
            
            owner = f"getlock thread {threading.get_native_id()}"
            logger.info(f"owner:{owner}, resource_id:{self.resource_id}")
            with self.resource_lock:
                lock_acquired = True

            owner = f"lock thread {threading.get_native_id()}"
            logger.info(f"owner:{owner}, resource_id:{self.resource_id}")
            if lock_acquired:
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
            else:
                # 如果无法获取锁，检查超时
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout while waiting for owner:{owner}, resource_id:{self.resource_id} lock: {self.resource_id}")
                time.sleep(0.1)  # 稍微等待后重试

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
        # 用来存储资源ID对应的锁
        self.lock_dict = {}

        # 创建一个 thread_local 对象，用来为每个线程分配独立的任务字典
        self._thread_local = threading.local()

        self.results = {}  # 使用字典存储每个eventId对应的结果
        self.lock = threading.Lock()  # 添加锁来同步对lock_dict的访问
        self.event_counter = 0  # 用于生成唯一的eventId
        self.event_counter_lock = threading.Lock()  # 用于保护event_counter的原子递增操作

    def _get_thread_tasks(self):
        """ 获取当前线程的任务字典 """
        if not hasattr(self._thread_local, 'tasks'):
            self._thread_local.tasks = {}  # 如果当前线程没有 tasks 属性，则初始化
        return self._thread_local.tasks

    def register_event(self, task_func, resource_id, kwargs=None):
        """ 注册事件并绑定资源ID，如果资源ID对应的锁已被占用，等待直到资源释放 """
        kwargs = kwargs or {}

        # 使用锁来同步对lock_dict的访问
        with self.lock:
            if resource_id not in self.lock_dict:
                self.lock_dict[resource_id] = threading.Lock()  # 为每个资源ID创建一个锁

        # 生成唯一的eventId
        event_id = self.generate_event_id()

        # 获取当前线程的任务字典
        tasks = self._get_thread_tasks()

        # 创建Iteratorize任务并传入对应的资源锁
        iteratorize_task = Iteratorize(task_func, resource_id, kwargs, resource_lock=self.lock_dict[resource_id])
        
        # 将任务存储到当前线程的任务字典中
        tasks[event_id] = iteratorize_task

        # 为该eventId初始化结果存储
        self.results[event_id] = []

        return event_id

    def execute_event_by_id(self, event_id):
        """ 根据eventId执行特定的事件 """
        # 获取当前线程的任务字典
        tasks = self._get_thread_tasks()
        
        task = tasks.get(event_id)
        if task is None:
            raise ValueError(f"Event with id {event_id} not found.")
     
        for result in task:
            self.results[event_id].append(result)
  
        # 任务执行完成后，删除任务
        self.clean_up_task(event_id)

    def execute_all_events(self):
        """ 执行当前线程注册的所有事件并收集结果 """
        # 获取当前线程的任务字典
        tasks = self._get_thread_tasks()
        
        # 将任务ID保存到一个列表中，避免在迭代时修改字典
        event_ids = list(tasks.keys())

        for event_id in event_ids:
            task = tasks.get(event_id)
            if task:
                # 执行任务
                for result in task:
                    self.results[event_id].append(result)
                
                # 执行完成后清理任务
                self.clean_up_task(event_id)

    def get_results(self, event_id=None):
        """ 获取某个任务的执行结果，如果没有传入event_id，返回所有任务的结果 """
        if event_id:
            if event_id not in self.results:
                raise ValueError(f"Results for event_id {event_id} not found.")
            return self.results[event_id]
        else:
            return self.results

    def generate_event_id(self):
        """ 生成唯一的eventId，使用锁来确保原子性 """
        with self.event_counter_lock:   
            self.event_counter += 1
            return f"event_{self.event_counter}"

    def clean_up_task(self, event_id):
        """ 清理已完成的任务 """
        # 获取当前线程的任务字典
        tasks = self._get_thread_tasks()
        
        if event_id in tasks:
            del tasks[event_id] 

    def clean_up_all_tasks(self):
        """ 清理所有已完成的任务，包括资源 """
        # 获取当前线程的任务字典
        tasks = self._get_thread_tasks()
        
        tasks.clear()
        self.results.clear()


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
 
    system_prompt = gpt_prompt_config.evaluate_system_prompt


    storage_context = StorageContext.from_defaults(
        persist_dir=f"/mnt/ceph/develop/jiawei/InterpretationoDreams/src/docs/doubao/60f9b7459a7749597e7efa71d1747bc4/storage/5a31ecd7-6ffe-4497-b4f7-cbbd113d922f"
    )

    def register_and_execute_events(event_manager, start_idx, end_idx):

        owner = f"start_event thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}")
        event_ids = []
        for i in range(start_idx, end_idx):
                
            owner = f"register_event thread {threading.get_native_id()}"
            logger.info(f"owner:{owner}")
            user_prompt = evaluate_system_prompt_template.format(
                problem=f"任务{i}",
                answer="结束" if i % 3 == 0 else "失败" if i % 3 == 1 else "等待中",
            )
            event_id = event_manager.register_event(
                task_function,
                resource_id=f"resource_{i % 2 + 1}",
                kwargs={
                    "llm_runable": llm_runable,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "storage_context": storage_context,
                },
            )

            event_ids.append(event_id)  # 将 event_id 存储到事件列表中
            
            owner = f"register_event end thread {threading.get_native_id()}"
            logger.info(f"owner:{owner}")

        # Execute event after registration
        event_manager.execute_all_events()

        owner = f"event_ids thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}")
        # Retrieve and print the result
        for event_id in event_ids:
            results = event_manager.get_results(event_id)
            print(f"Final Results for Event {event_id}: {results}")

    # Thread 1 - Registers and executes events from 0 to 4
    thread1 = threading.Thread(target=register_and_execute_events, args=(event_manager, 0, 5))

    # Thread 2 - Registers and executes events from 5 to 9
    thread2 = threading.Thread(target=register_and_execute_events, args=(event_manager, 5, 10))

    # Start both threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    print("All events processed successfully.")

if __name__ == "__main__":
    main()