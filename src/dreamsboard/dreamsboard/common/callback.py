import asyncio
import atexit
import logging
import queue
import threading
import time
import traceback
from multiprocessing import Lock, Process, Queue
from threading import Thread

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

    def __init__(self, func, resource_id, sentinel, kwargs={}, resource_lock=None):
        self.mfunc = func
        self.q = Queue()
        self.sentinel = sentinel
        self.kwargs = kwargs
        self.stop_now = False
        self.resource_id = resource_id
        self.resource_lock = resource_lock  # 传入具体的锁，而不是整个lock_dict

        self._callback = self._callback_func()  # 保存回调函数
        self.thread = Thread(target=self._run)  # 延迟创建线程
        self.thread.start()

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

        owner = f"getlock thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}, resource_id:{self.resource_id}")

        self.resource_lock.acquire()
        owner = f"lock thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}, resource_id:{self.resource_id}")

        try:
            # 执行任务，传入回调函数
            self.mfunc(
                callback=self._callback, resource_id=self.resource_id, **self.kwargs
            )
        except ValueError:
            traceback.print_exc()
        except:
            traceback.print_exc()
        finally:
            self.resource_lock.release()
            # 任务结束，放入sentinel来标识结束
            self.q.put(self.sentinel)

    def __iter__(self):
        """控制迭代器，第一次迭代时启动线程"""
        return self

    def __next__(self):
        """Fetch the next value from the queue or stop if sentinel is received."""
        obj = self.q.get(True, None)
        if obj == self.sentinel:  # 使用 == 比较
            raise StopIteration
        else:
            return obj

    def __del__(self):
        """Cleanup resources, stop the task if needed."""
        self.stop_now = True
        if self.thread.is_alive():
            self.thread.join()

    def __enter__(self):
        """Used for context management."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the iteration when exiting context."""
        self.stop_now = True
        if self.thread.is_alive():
            self.thread.join()


class ProcessIteratorize:
    """
    这个类会将一个任务封装为延迟执行的迭代器（Generator）。 
    """

    def __init__(self, func, resource_id, sentinel, kwargs={}):
        self.mfunc = func
        self.q = Queue()
        self.sentinel = sentinel
        self.kwargs = kwargs
        self.stop_now = False
        self.resource_id = resource_id 

        self._callback = self._callback_func()  # 保存回调函数
        self.process = Process(target=self._run)  # 延迟创建进程
        self.process.start()
        atexit.register(self.cleanup)  # Jupyter 退出时清理进程

    def _callback_func(self):
        """回调函数用于收集结果，并控制迭代的结束"""

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        return _callback

    def _run(self):
        """在迭代开始时启动线程并执行任务 
        """

        owner = f"getlock thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}, resource_id:{self.resource_id}")


        try: 
            owner = f"lock thread {threading.get_native_id()}"
            logger.info(f"owner:{owner}, resource_id:{self.resource_id}")
            # 执行任务，传入回调函数
            self.mfunc(
                callback=self._callback, resource_id=self.resource_id, **self.kwargs
            )
        except ValueError:
            traceback.print_exc()
        except:
            traceback.print_exc()
        finally: 
            # 任务结束，放入sentinel来标识结束
            self.q.put(self.sentinel)

    def __iter__(self):
        """控制迭代器，第一次迭代时启动线程"""
        return self

    def __next__(self):
        """Fetch the next value from the queue or stop if sentinel is received."""
        obj = self.q.get(True, None)
        if obj == self.sentinel:  # 使用 == 比较
            raise StopIteration
        else:
            return obj

    def cleanup(self):
        self.stop_now = True
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()

    def __del__(self):
        """Cleanup resources, stop the task if needed."""
        pass

    def __enter__(self):
        """Used for context management."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the iteration when exiting context."""
        self.stop_now = True
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


def call_func(func, resource_id, kwargs={}):
 
    sentinel = "SENTINEL"
    result_holder = []

    # 创建多个 Iteratorize 实例，模拟多个并发任务
    it = ProcessIteratorize(
        func=func,
        resource_id=resource_id,
        sentinel=sentinel, 
        kwargs=kwargs,
    )
    thread_id = threading.get_ident()

    # 确保当前线程在注册时有一个列表
    process_registry.setdefault(thread_id, [])
    process_registry[thread_id].append(it.process)
    thread = threading.Thread(target=run_iterator, args=(it, result_holder))

    thread.start()
    thread.join()

    return result_holder


# 运行 Iteratorize 并获取结果
def run_iterator(it, result_holder):
    with it as iterator:
        for result in iterator:
            print(f"Received: {result}")
            result_holder.append(result)
        print(f"Iteration complete for {it.resource_id}.")


class EventManager:
    def __init__(self):
        self.lock_dict = {}  # 用来存储资源ID对应的锁
        self.tasks = queue.Queue()  # 使用Queue来存储事件任务
        self.results = {}  # 使用字典存储每个eventId对应的结果
        self.lock = threading.Lock()  # 添加锁来同步对lock_dict的访问
        self.event_counter = 0  # 用于生成唯一的eventId
        self.event_counter_lock = (
            threading.RLock()
        )  # 用于保护event_counter的原子递增操作
        self.execution_thread = threading.Thread(
            target=self._execute_task_queue, daemon=True
        )  # 使用 daemon 线程
        self.execution_thread.start()  # 启动任务执行线程

    def register_event(self, task_func, resource_id, kwargs=None):
        """注册事件并绑定资源ID，如果资源ID对应的锁已被占用，等待直到资源释放"""
        kwargs = kwargs or {}

        # 使用锁来同步对lock_dict的访问
        try:
            self.lock.acquire()
            if resource_id not in self.lock_dict:
                self.lock_dict[resource_id] = threading.Lock()  # 为每个资源ID创建一个锁
        finally:
            self.lock.release()

        # 生成唯一的eventId
        event_id = self.generate_event_id(resource_id)
        sentinel = "SENTINEL"
        # 创建Iteratorize任务并传入对应的资源锁
        iteratorize_task = Iteratorize(
            task_func,
            resource_id,
            sentinel=sentinel,
            resource_lock=self.lock_dict[resource_id],
            kwargs=kwargs,
        )

        # 将任务存储到队列中
        self.tasks.put(
            (event_id, iteratorize_task)
        )  # 将event_id和task作为元组放入队列中

        # 为该eventId初始化结果存储
        self.results[event_id] = []

        return event_id

    def _execute_task_queue(self):
        """使用asyncio异步处理任务队列"""

        while True:
            # 从队列中取任务，如果队列为空则会阻塞，直到有新的任务加入
            event_id, task = self.tasks.get()

            if task:
                for result in task:
                    self.results[event_id].append(result)

    def get_results(self, event_id=None):
        """获取某个任务的执行结果，如果没有传入event_id，返回所有任务的结果"""
        if event_id in self.results:
            return self.results[event_id]
        else:
            return self.results

    def generate_event_id(self, resource_id):
        """生成唯一的事件ID，格式：event:{resource_id}:{event_counter}"""
        try:
            self.event_counter_lock.acquire()
            self.event_counter += 1
            return f"event:{resource_id}:{self.event_counter}"
        finally:
            self.event_counter_lock.release()

    def clean_up_all_tasks(self):
        """清理所有已完成的任务包括资源"""
        self.results.clear()


# 用于保存各线程的子进程引用（示例：全局字典）
process_registry = {}
event_manager = EventManager()
