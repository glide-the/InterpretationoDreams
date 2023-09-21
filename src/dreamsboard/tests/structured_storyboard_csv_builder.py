from dreamsboard.chains.base import StoryBoardDreamsGenerationChain
from dreamsboard.document_loaders.csv_structured_storyboard_loader import StructuredStoryboardCSVBuilder

from langchain.chat_models import ChatOpenAI
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


def test_structured_storyboard_csv_builder() -> None:

    llm = ChatOpenAI(
        streaming=False,
        verbose=True,
        openai_api_key="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",
        openai_api_base="https://api.openai.com/v1",
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        # openai_proxy="http://127.0.0.1:7890"
    )
    dreams_generation_chain = StoryBoardDreamsGenerationChain.from_dreams_personality_chain(
        llm=llm, csv_file_path="/media/checkpoint/speech_data/抖音作品/ieAeWyXU/str/ieAeWyXU_keyframe.csv")

    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

    # wandb documentation to configure wandb using env variables
    # https://docs.wandb.ai/guides/track/advanced/environment-variables
    # here we are configuring the wandb project name
    os.environ["WANDB_PROJECT"] = "StoryBoardDreamsGenerationChain"
    os.environ["WANDB_API_KEY"] = "974207f7173417ef95d2ebad4cbe7f2f9668a093"
    from langchain.callbacks import wandb_tracing_enabled

    output = dreams_generation_chain.run()
    logger.info(output)
