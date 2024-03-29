import logging

from langchain.chat_models import ChatOpenAI

from dreamsboard.dreams.builder_cosplay_code.base import StructuredDreamsStoryboard
from dreamsboard.dreams.dreams_personality_chain.base import StoryBoardDreamsGenerationChain
import langchain

from dreamsboard.engine.generate.code_generate import QueryProgramGenerator, EngineProgramGenerator, AIProgramGenerator
from dreamsboard.engine.loading import load_store_from_storage
from dreamsboard.engine.storage.storage_context import StorageContext

langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def test_structured_dreams_storyboard_store_print() -> None:
    storage_context = StorageContext.from_defaults(persist_dir="./storage_ieAjskDr_keyframe")

    logger.info(    storage_context.dreams_analysis_store)
    assert True
