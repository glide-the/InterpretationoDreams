import logging

import langchain
from langchain_community.chat_models import ChatOpenAI

from dreamsboard.dreams.builder_cosplay_code.base import StructuredDreamsStoryboard
from dreamsboard.dreams.dreams_personality_chain.base import (
    StoryBoardDreamsGenerationChain,
)
from dreamsboard.engine.generate.code_generate import (
    AIProgramGenerator,
    EngineProgramGenerator,
    QueryProgramGenerator,
)
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
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        code_gen_builder = load_store_from_storage(storage_context)
        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:
        llm = ChatOpenAI(verbose=True)

        dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
            llm_runable=llm,
            csv_file_path="/media/checkpoint/speech_data/抖音作品/id46Bv3g/str/id46Bv3g_keyframe.csv",
        )

        output = dreams_generation_chain.run()
        logger.info("dreams_guidance_context:" + output.get("dreams_guidance_context"))
        logger.info(
            "dreams_personality_context:" + output.get("dreams_personality_context")
        )
        dreams_guidance_context = output.get("dreams_guidance_context")
        dreams_personality_context = output.get("dreams_personality_context")

        storyboard_executor = StructuredDreamsStoryboard.form_builder(
            llm_runable=llm,
            builder=dreams_generation_chain.builder,
            dreams_guidance_context=dreams_guidance_context,
            dreams_personality_context=dreams_personality_context,
        )
        code_gen_builder = storyboard_executor.loader_cosplay_builder()

    executor = code_gen_builder.build_executor()
    logger.info(executor.executor_code)
    assert True
