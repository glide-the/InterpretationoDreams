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
        cosplay_role="ieAjabk1",
        source_url="./ieAjabk1_keyframe",
        keyframe="./ieAjabk1_keyframe",
        keyframe_path="./ieAjabk1_keyframe",
        storage_keyframe="./ieAjabk1_keyframe",
        storage_keyframe_path="./ieAjabk1_keyframe",
    )
    out = analysis.format_md()
    print(out.text)
