from dreamsboard.engine.constants import DATA_KEY, TYPE_KEY
from dreamsboard.engine.schema import (
    BaseNode,
)

from dreamsboard.engine.generate.code_generate import (
    CodeGenerator,
    BaseProgramGenerator,
    QueryProgramGenerator,
    AIProgramGenerator,
    EngineProgramGenerator,
)


def template_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_template(template_dict: dict) -> BaseNode:
    template_type = template_dict[TYPE_KEY]
    data_dict = template_dict[DATA_KEY]
    doc: BaseNode

    if "extra_info" in data_dict:
        return legacy_json_to_template(template_dict)
    else:
        if template_type == BaseProgramGenerator.get_type():
            doc = BaseProgramGenerator.model_validate(data_dict)
        elif template_type == QueryProgramGenerator.get_type():
            doc = QueryProgramGenerator.model_validate(data_dict)
        elif template_type == AIProgramGenerator.get_type():
            doc = AIProgramGenerator.model_validate(data_dict)
        elif template_type == EngineProgramGenerator.get_type():
            doc = EngineProgramGenerator.model_validate(data_dict)
        else:
            raise ValueError(f"Unknown doc type: {template_type}")

        return doc


def legacy_json_to_template(template_dict: dict) -> CodeGenerator:
    """Todo: Deprecated legacy support for old node versions."""
    template_type = template_dict[TYPE_KEY]
    data_dict = template_dict[DATA_KEY]
    generator: CodeGenerator

    text = data_dict.get("text", "")
    _base_render_data = data_dict.get("render_data", {}) or {}
    id_ = data_dict.get("doc_id", None)

    if template_type == BaseProgramGenerator.get_type():
        generator = BaseProgramGenerator.from_config(cfg={
            "code_file": "base_template.py-tpl",
            "render_data": _base_render_data,
        })
    elif template_type == QueryProgramGenerator.get_type():
        generator = QueryProgramGenerator.from_config(cfg={
            "query_code_file": "dreams_query_template.py-tpl",
            "render_data": _base_render_data,
        })
    elif template_type == AIProgramGenerator.get_type():
        generator = AIProgramGenerator.from_config(cfg={
            "ai_code_file": "ai_template.py-tpl",
            "render_data": _base_render_data,
        })
    elif template_type == EngineProgramGenerator.get_type():
        generator = EngineProgramGenerator.from_config(cfg={
            "engine_code_file": "engine_template.py-tpl",
            "render_data": _base_render_data,
        })
    else:
        raise ValueError(f"Unknown template type: {template_type}")

    return generator
