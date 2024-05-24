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
        cosplay_role="澈洌",
        source_url="https://v.douyin.com/ieAkpNXB/",
        keyframe="ieAkpNXB_keyframe.csv",
        keyframe_path="./ieAkpNXB_keyframe.csv",
        storage_keyframe="storage_ieAkpNXB_keyframe",
        storage_keyframe_path="./storage_ieAkpNXB_keyframe",
    )
    out = analysis.write_md(output_path="./03_报备式恋爱请查收_澈洌.md")
    print(out.text)
