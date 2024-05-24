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
        cosplay_role="辰夕",
        source_url="https://v.douyin.com/ieAjskDr/",
        keyframe="ieAjskDr_keyframe.csv",
        keyframe_path="./ieAjskDr_keyframe.csv",
        storage_keyframe="storage_ieAjskDr_keyframe",
        storage_keyframe_path="./storage_ieAjskDr_keyframe",
    )

    out = analysis.write_md(output_path="./04_心里闷闷的可能就是有点想你_辰夕.md")
    print(out.text)
