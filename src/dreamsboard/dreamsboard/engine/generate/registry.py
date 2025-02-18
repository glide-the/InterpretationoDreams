from typing import Dict, Type

from dreamsboard.engine.generate.code_generate import (
    AIProgramGenerator,
    BaseProgramGenerator,
    CodeGenerator,
    EngineProgramGenerator,
    QueryProgramGenerator,
)
from dreamsboard.engine.schema import ObjectTemplateType

TEMPLATE_TYPE_TO_GENERATOR_CLASS: Dict[ObjectTemplateType, Type[CodeGenerator]] = {
    ObjectTemplateType.BaseProgramGenerator: BaseProgramGenerator,
    ObjectTemplateType.QueryProgramGenerator: QueryProgramGenerator,
    ObjectTemplateType.AIProgramGenerator: AIProgramGenerator,
    ObjectTemplateType.EngineProgramGenerator: EngineProgramGenerator,
}
