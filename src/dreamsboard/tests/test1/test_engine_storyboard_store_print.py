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
        source_url="https://v.douyin.com/iRMa9DMW",
        keyframe="iRMa9DMW_keyframe.csv",
        keyframe_path="iRMa9DMW_keyframe.csv",
        storage_keyframe="storage_iRMa9DMW_keyframe",
        storage_keyframe_path="./storage_iRMa9DMW_keyframe",
    )
    out = analysis.write_md(
        output_path="./01_宝今天煮饺子把皮煮开了原来是喜欢你露馅儿了_阿七.md"
    )
    print(out.text)
