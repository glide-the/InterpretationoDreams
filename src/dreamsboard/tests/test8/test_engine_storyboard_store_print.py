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
        source_url="https://v.douyin.com/id46Bv3g/",
        keyframe="id46Bv3g_keyframe.csv",
        keyframe_path="./id46Bv3g_keyframe.csv",
        storage_keyframe="storage_id46Bv3g_keyframe",
        storage_keyframe_path="./storage_id46Bv3g_keyframe",
    )

    out = analysis.write_md(output_path="./08_78块的冰淇淋掉地上了呜呜呜呜_阿七.md")
    print(out.text)
