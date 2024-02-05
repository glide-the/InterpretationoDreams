from dreamsboard.document_loaders.csv_structured_storyboard_loader import StructuredStoryboardCSVBuilder

import logging
import langchain

langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


def test_structured_storyboard_csv_builder() -> None:
    builder = StructuredStoryboardCSVBuilder(
        csv_file_path="/media/gpt4-pdf-chatbot-langchain/InterpretationoDreams/社会交流步骤分析/msg_extract_csv/msg-1_1_keyframe.csv")
    builder.load()  # 替换为你的CSV文件路径
    selected_columns = ["story_board_role", "story_board_text", "story_board"]
    formatted_text = builder.build_text(selected_columns)
    logger.info("formatted_text:" + formatted_text)
    assert formatted_text is not None


def test_structured_storyboard_csv_builder_export_role() -> None:
    builder = StructuredStoryboardCSVBuilder(
        csv_file_path="/media/gpt4-pdf-chatbot-langchain/InterpretationoDreams/社会交流步骤分析/msg_extract_csv/msg-1_1_keyframe.csv")
    builder.load()  # 替换为你的CSV文件路径
    export_role = builder.export_role()
    logger.info(export_role)
    assert export_role is not None


def test_structured_storyboard_csv_builder_msg() -> None:
    builder = StructuredStoryboardCSVBuilder(csv_file_path="/media/gpt4-pdf-chatbot-langchain/InterpretationoDreams/社会交流步骤分析/msg_extract_csv/msg-1_5_keyframe.csv")
    builder.load()  # 替换为你的CSV文件路径

    formatted_text = builder.build_msg()
    logger.info("formatted_text:" + formatted_text)
    assert formatted_text is not None
