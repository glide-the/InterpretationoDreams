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
        cosplay_role="今天要吃三碗饭",
        source_url="https://v.douyin.com/ieDRHjmD/",
        keyframe="ieDRHjmD_keyframe.csv",
        keyframe_path="./ieDRHjmD_keyframe.csv",
        storage_keyframe="storage_ieDRHjmD_keyframe",
        storage_keyframe_path="./storage_ieDRHjmD_keyframe",
    )
    out = analysis.write_md(output_path="./02_尊嘟假嘟呀_今天要吃三碗饭.md")
    print(out.text)
