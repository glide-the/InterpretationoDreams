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
    builder = StructuredStoryboardCSVBuilder(csv_file_path="/media/checkpoint/speech_data/抖音作品/ieAeWyXU/str"
                                                           "/ieAeWyXU_keyframe.csv")
    builder.load()  # 替换为你的CSV文件路径
    selected_columns = ["story_board_text", "story_board"]
    formatted_text = builder.build_text(selected_columns)
    logger.info("formatted_text:"+formatted_text)
    assert formatted_text is not None
