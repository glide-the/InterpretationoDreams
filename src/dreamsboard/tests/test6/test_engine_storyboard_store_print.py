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
        cosplay_role="兔兔没有牙",
        source_url="https://v.douyin.com/ieAeWyXU/",
        keyframe="ieAeWyXU_keyframe.csv",
        keyframe_path="./ieAeWyXU_keyframe.csv",
        storage_keyframe="storage_ieAeWyXU_keyframe",
        storage_keyframe_path="./storage_ieAeWyXU_keyframe",
    )
    out = analysis.write_md(output_path="./06_今日的温柔报告_兔兔没有牙.md")
    print(out.text)
