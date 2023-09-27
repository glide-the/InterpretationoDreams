import logging

from langchain.chat_models import ChatOpenAI

from dreamsboard.chains.base import StructuredDreamsStoryboard, StoryBoardDreamsGenerationChain
from dreamsboard.generate.code_generate import BaseProgramGenerator, EngineProgramGenerator, QueryProgramGenerator
from dreamsboard.generate.run_generate import CodeGeneratorBuilder
import langchain
import os
langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def test_structured_dreams_storyboard() -> None:
    llm = ChatOpenAI(
        verbose=True
    )

    dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
        llm=llm, csv_file_path="/media/checkpoint/speech_data/抖音作品/ieAeWyXU/str/ieAeWyXU_keyframe.csv")

    output = dreams_generation_chain.run()
    logger.info("dreams_guidance_context:" + output.get("dreams_guidance_context"))
    logger.info("dreams_personality_context:" + output.get("dreams_personality_context"))
    dreams_guidance_context = output.get("dreams_guidance_context")
    dreams_personality_context = output.get("dreams_personality_context")

    storyboard_executor = StructuredDreamsStoryboard.form_builder(llm=llm,
                                                                  builder=dreams_generation_chain.builder,
                                                                  dreams_guidance_context=dreams_guidance_context,
                                                                  dreams_personality_context=dreams_personality_context
                                                                  )
    code_gen_builder = storyboard_executor.loader_cosplay_builder()

    logger.info(code_gen_builder)
    assert True
