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
        source_url="https://v.douyin.com/ieA2E1F7/",
        keyframe="ieA2E1F7_keyframe.csv",
        keyframe_path="./ieA2E1F7_keyframe.csv",
        storage_keyframe="storage_ieA2E1F7_keyframe",
        storage_keyframe_path="./storage_ieA2E1F7_keyframe",
    )

    out = analysis.write_md(output_path="./07_报告今天发工资啦啦啦啦_阿七.md")
    print(out.text)
