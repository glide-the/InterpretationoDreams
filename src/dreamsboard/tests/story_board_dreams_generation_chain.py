from langchain.chat_models import ChatOpenAI
from dreamsboard.chains.base import StoryBoardDreamsGenerationChain
import logging
import langchain
import os
langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


def test_story_board_dreams_generation_chain():

    # os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

    # wandb documentation to configure wandb using env variables
    # https://docs.wandb.ai/guides/track/advanced/environment-variables
    # here we are configuring the wandb project name
    # os.environ["WANDB_PROJECT"] = "StoryBoardDreamsGenerationChain"
    # os.environ["WANDB_API_KEY"] = "key"
    llm = ChatOpenAI(
        verbose=True
    )

    dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
        llm=llm, csv_file_path="/media/checkpoint/speech_data/抖音作品/ieA2m5p2/str/ieA2m5p2_keyframe.csv")

    output = dreams_generation_chain.run()
    logger.info("dreams_guidance_context:"+output.get("dreams_guidance_context"))
    logger.info("dreams_personality_context:"+output.get("dreams_personality_context"))
    assert True
