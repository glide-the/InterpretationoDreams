from dreamsboard.engine.constants import DATA_KEY, TYPE_KEY
from dreamsboard.engine.dreams_personality.dreams_personality import DreamsPersonalityNode
from dreamsboard.engine.schema import (
    BaseNode,
)


def analysis_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_analysis(analysis_dict: dict) -> BaseNode:
    analysis_type = analysis_dict[TYPE_KEY]
    data_dict = analysis_dict[DATA_KEY]
    doc: BaseNode

    if "extra_info" in data_dict:
        return legacy_json_to_analysis(analysis_dict)
    else:
        if analysis_type == DreamsPersonalityNode.get_type():
            doc = DreamsPersonalityNode.model_validate(data_dict)
        else:
            raise ValueError(f"Unknown doc type: {analysis_type}")

        return doc


def legacy_json_to_analysis(analysis_dict: dict) -> DreamsPersonalityNode:
    """Todo: Deprecated legacy support for old node versions."""
    analysis_type = analysis_dict[TYPE_KEY]
    data_dict = analysis_dict[DATA_KEY]
    generator: DreamsPersonalityNode

    _base_render_data = data_dict.get("render_data", {}) or {}

    if analysis_type == DreamsPersonalityNode.get_type():
        generator = DreamsPersonalityNode.from_config(cfg={
            "code_file": "base_analysis.py-tpl",
            "render_data": _base_render_data,
        })
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    return generator
