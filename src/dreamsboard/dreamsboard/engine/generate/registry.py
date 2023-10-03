from typing import Type, Dict

from dreamsboard.engine.generate.code_generate import (
    CodeGenerator,
    BaseProgramGenerator,
    QueryProgramGenerator,
    AIProgramGenerator,
    EngineProgramGenerator,
)
from dreamsboard.engine.schema import ObjectTemplateType


TEMPLATE_TYPE_TO_GENERATOR_CLASS: Dict[ObjectTemplateType, Type[CodeGenerator]] = {
    ObjectTemplateType.BaseProgramGenerator: BaseProgramGenerator,
    ObjectTemplateType.QueryProgramGenerator: QueryProgramGenerator,
    ObjectTemplateType.AIProgramGenerator: AIProgramGenerator,
    ObjectTemplateType.EngineProgramGenerator: EngineProgramGenerator,
}
