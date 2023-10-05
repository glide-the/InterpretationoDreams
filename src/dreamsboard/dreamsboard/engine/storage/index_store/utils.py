from dreamsboard.engine.constants import DATA_KEY, TYPE_KEY
from dreamsboard.engine.data_structs.data_structs import IndexStruct
from dreamsboard.engine.generate.registry import TEMPLATE_TYPE_TO_GENERATOR_CLASS


def index_struct_to_json(index_struct: IndexStruct) -> dict:
    index_struct_dict = {
        TYPE_KEY: index_struct.get_type(),
        DATA_KEY: index_struct.to_json(),
    }
    return index_struct_dict


def json_to_index_struct(struct_dict: dict) -> IndexStruct:
    type = struct_dict[TYPE_KEY]
    data_dict = struct_dict[DATA_KEY]
    cls = TEMPLATE_TYPE_TO_GENERATOR_CLASS[type]
    try:
        return cls.from_json(data_dict)
    except TypeError:
        return cls.from_dict(data_dict)
