"""Index registry."""

from typing import Dict, Type

from dreamsboard.engine.data_structs.struct_type import IndexStructType
from dreamsboard.engine.engine_builder import BaseEngineBuilder, CodeGeneratorBuilder

INDEX_STRUCT_TYPE_TO_INDEX_CLASS: Dict[IndexStructType, Type[BaseEngineBuilder]] = {
    IndexStructType.CODE_GENERATOR: CodeGeneratorBuilder,
}
