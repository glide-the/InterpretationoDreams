from dreamsboard.engine.entity.dreams_personality.dreams_personality import DreamsPersonalityNode
from dreamsboard.engine.storage.dreams_analysis_store.simple_dreams_analysis_store import SimpleDreamsAnalysisStore
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


def test_simple_store_from_persist_path():
    store = SimpleDreamsAnalysisStore.from_persist_dir(persist_dir="./storage")

    assert store is not None


def test_simple_store():
    store = SimpleDreamsAnalysisStore()
    analysis_all = store.analysis_all
    logger.info(analysis_all)
    assert store is not None


def test_simple_store_add():
    store = SimpleDreamsAnalysisStore()
    dreams = DreamsPersonalityNode.from_config(cfg={
        "dreams_guidance_context": "dreams_guidance",
        "dreams_personality_context": "dreams_personality"
    })
    store.add_analysis([dreams])
    logger.info(store.analysis_all)
    assert store is not None


def test_simple_store_save():
    store = SimpleDreamsAnalysisStore()
    dreams = DreamsPersonalityNode.from_config(cfg={
        "dreams_guidance_context": "dreams_guidance",
        "dreams_personality_context": "dreams_personality"
    })
    store.add_analysis([dreams])
    logger.info(store.analysis_all)
    dreams_analysis_store_path = concat_dirs(dirname="./storage", basename="dreams_analysis_store.json")
    store.persist(persist_path=dreams_analysis_store_path)
    assert store is not None


def test_simple_store_load():
    store = SimpleDreamsAnalysisStore.from_persist_dir(persist_dir="./storage")

    logger.info(store.analysis_all)

    assert store is not None
