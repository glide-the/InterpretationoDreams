import logging

from dreamsboard.dreams.coplay_analysis_md.base import CosplayAnalysisMD

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def test_structured_dreams_storyboard_store_print() -> None:
    analysis = CosplayAnalysisMD(
        cosplay_role="阿七",
        source_url="https://v.douyin.com/ieAjabk1/",
        keyframe="ieAjabk1_keyframe.csv",
        keyframe_path="./ieAjabk1_keyframe.csv",
        storage_keyframe="./storage_ieAjabk1_keyframe",
        storage_keyframe_path="./storage_ieAjabk1_keyframe",
    )
    out = analysis.write_md(output_path="./05_报告今天是香香的姐姐_阿七.md")
    print(out.text)