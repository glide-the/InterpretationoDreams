import threading
import time
from multiprocessing import Lock

from dreamsboard.common.callback import Iteratorize


# 假设的任务函数，执行一段时间并回调结果
def sample_task(callback, resource_id, **kwargs):
    print(f"Task for resource {resource_id} started...")
    for i in range(5):
        time.sleep(1)
        callback(f"Task {resource_id} result {i}")


# 测试 Iteratorize 类
def test_iteratorize():
    # 创建一个资源锁，用于保护资源
    resource_lock = Lock()
    sentinel = "SENTINEL"
    # 创建 Iteratorize 实例，传入任务函数、资源ID、资源锁
    it = Iteratorize(
        func=sample_task,
        resource_id="Resource_1",
        sentinel=sentinel,
        resource_lock=resource_lock,
    )

    # 使用迭代器来获取任务的结果
    with it as iterator:
        for result in iterator:
            print(f"Received: {result}")
        print("Iteration complete.")


# 测试 Iteratorize 类
def test_iteratorize_threads():
    # 创建一个资源锁，用于保护资源
    resource_lock = Lock()
    sentinel = "SENTINEL"

    # 创建多个 Iteratorize 实例，模拟多个并发任务
    iterators = [
        Iteratorize(
            func=sample_task,
            resource_id=f"Resource_{i}",
            sentinel=sentinel,
            resource_lock=resource_lock,
        )
        for i in range(3)  # 创建 3 个 Iteratorize 实例
    ]

    # 启动所有任务
    threads = []
    for it in iterators:
        thread = threading.Thread(target=run_iterator, args=(it,))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("All tasks completed.")


# 运行 Iteratorize 并打印结果
def run_iterator(it):
    with it as iterator:
        for result in iterator:
            print(f"Received: {result}")
        print(f"Iteration complete for {it.resource_id}.")
