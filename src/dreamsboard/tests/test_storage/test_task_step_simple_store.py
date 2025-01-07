from dreamsboard.engine.entity.task_step.task_step import TaskStepNode
from dreamsboard.engine.storage.task_step_store.simple_task_step_store import SimpleTaskStepStore
import logging
import langchain

from dreamsboard.engine.utils import concat_dirs

langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def test_simple_task_step_store_from_persist_path():
    store = SimpleTaskStepStore.from_persist_dir(persist_dir="./storage")

    assert store is not None


def test_simple_task_step_store():
    store = SimpleTaskStepStore()
    task_step_all = store.task_step_all
    logger.info(task_step_all)
    assert store is not None


def test_simple_task_step_store_add():
    store = SimpleTaskStepStore()
    task_step = TaskStepNode.from_config(cfg={
        "start_task_context": "start_task_context",
        "aemo_representation_context": "aemo_representation_context",
        "task_step_name": "task_step_name",
        "task_step_description": "task_step_description",
        "task_step_level": "task_step_level"
    })
    store.add_task_step([task_step])
    logger.info(store.task_step_all)
    assert store is not None


def test_simple_task_step_store_save():
    store = SimpleTaskStepStore()
    task_step = TaskStepNode.from_config(cfg={
        "start_task_context": "start_task_context",
        "aemo_representation_context": "aemo_representation_context",
        "task_step_name": "task_step_name",
        "task_step_description": "task_step_description",
        "task_step_level": "task_step_level"
    })
    store.add_task_step([task_step])
    logger.info(store.task_step_all)
    task_step_store_path = concat_dirs(dirname="./storage", basename="task_step_store.json")
    store.persist(persist_path=task_step_store_path)
    assert store is not None


def test_simple_task_step_store_load():

    store = SimpleTaskStepStore.from_persist_dir(persist_dir="./storage")
    logger.info(store.task_step_all)

    assert store is not None
