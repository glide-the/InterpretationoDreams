import json
import logging
from typing import Callable

import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")
CrossEncoder = sentence_transformers.CrossEncoder

from dreamsboard.document_loaders.csv_structured_storyboard_loader import (
    StructuredStoryboardCSVBuilder,
)
from dreamsboard.dreams.task_step_to_question_chain.base import TaskStepToQuestionChain
from dreamsboard.dreams.task_step_to_question_chain.weaviate.prepare_load import (
    get_query_hash,
)
from dreamsboard.engine.storage.task_step_store.simple_task_step_store import (
    SimpleTaskStepStore,
)
from dreamsboard.engine.storage.task_step_store.types import DEFAULT_PERSIST_FNAME
from dreamsboard.engine.utils import concat_dirs
from dreamsboard.vector.faiss_kb_service import FaissCollectionService

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)

"""
#### 场景加载模块

编写符合计算机科学领域的 故事情境提示词，生成研究情境（story_scenario_context），替换现有的langchain会话模板，
对每个子任务指令转换为子问题
召回问题前3条,存入task_step_question_context
调用llm，生成task_step_question_answer
"""
@pytest.fixture
def llm(chat_openai_factory: Callable[..., object]) -> object:
    """Provide a deterministic ChatOpenAI-compatible client."""

    return chat_openai_factory(temperature=0.1, top_p=0.9, verbose=True)


@pytest.fixture
def task_step_store() -> SimpleTaskStepStore:
    """Load the persisted task step store for integration tests."""

    return SimpleTaskStepStore.from_persist_dir(persist_dir="./storage")


@pytest.fixture(scope="session")
def cross_encoder(cross_encoder_path: str) -> CrossEncoder:
    """Instantiate the cross-encoder model declared for integration tests."""

    return CrossEncoder(
        cross_encoder_path,
        automodel_args={"torch_dtype": "auto"},
        trust_remote_code=True,
    )


@pytest.fixture
def collection(start_task_context: str, embed_model_path: str) -> FaissCollectionService:
    """Return a FAISS collection service bound to the shared task context."""

    collection_id = get_query_hash(start_task_context)
    collection = FaissCollectionService(
        kb_name=collection_id,
        embed_model=embed_model_path,
        vector_name="samples",
        device="cpu",
    )
    yield collection
    collection.do_clear_vs()


def _build_chain(
    llm: object,
    start_task_context: str,
    task_step_store: SimpleTaskStepStore,
    collection: FaissCollectionService,
    cross_encoder: CrossEncoder,
) -> TaskStepToQuestionChain:
    return TaskStepToQuestionChain.from_task_step_to_question_chain(
        base_path="./",
        llm_runable=llm,
        start_task_context=start_task_context,
        task_step_store=task_step_store,
        collection=collection,
        cross_encoder=cross_encoder,
    )


def test_invoke_task_step_to_question(
    llm: object,
    start_task_context: str,
    task_step_store: SimpleTaskStepStore,
    collection: FaissCollectionService,
    cross_encoder: CrossEncoder,
):
    task_step_to_question_chain = _build_chain(
        llm, start_task_context, task_step_store, collection, cross_encoder
    )

    task_step_id = list(task_step_store.task_step_all.keys())[0]
    task_step_to_question_chain.invoke_task_step_to_question(task_step_id)
    assert task_step_store.task_step_all is not None

    task_step_store_path = concat_dirs(
        dirname="./storage", basename=DEFAULT_PERSIST_FNAME
    )
    task_step_store.persist(persist_path=task_step_store_path)


def test_invoke_task_step_question_context(
    llm: object,
    start_task_context: str,
    task_step_store: SimpleTaskStepStore,
    collection: FaissCollectionService,
    cross_encoder: CrossEncoder,
):
    task_step_to_question_chain = _build_chain(
        llm, start_task_context, task_step_store, collection, cross_encoder
    )
    task_step_id = list(task_step_store.task_step_all.keys())[0]
    task_step_to_question_chain.invoke_task_step_question_context(task_step_id)
    assert task_step_store.task_step_all is not None
    task_step_store_path = concat_dirs(
        dirname="./storage", basename=DEFAULT_PERSIST_FNAME
    )
    task_step_store.persist(persist_path=task_step_store_path)


def test_export_csv_file_path(
    llm: object,
    start_task_context: str,
    task_step_store: SimpleTaskStepStore,
    collection: FaissCollectionService,
    cross_encoder: CrossEncoder,
):
    task_step_to_question_chain = _build_chain(
        llm, start_task_context, task_step_store, collection, cross_encoder
    )

    task_step_id = list(task_step_store.task_step_all.keys())[0]
    csv_file_path = task_step_to_question_chain.export_csv_file_path(task_step_id)
    logger.info("csv_file_path:" + csv_file_path)
    assert csv_file_path is not None

    builder = StructuredStoryboardCSVBuilder(csv_file_path=csv_file_path)
    builder.load()
    selected_columns = ["story_board_role", "story_board_text", "story_board"]
    formatted_text = builder.build_text(task_step_id, selected_columns)
    logger.info("formatted_text:" + formatted_text)
    assert formatted_text is not None
