from dreamsboard.engine.constants import DATA_KEY, TYPE_KEY
from dreamsboard.engine.entity.task_step.task_step import TaskStepNode
from dreamsboard.engine.schema import (
    BaseNode,
)


def analysis_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_analysis(analysis_dict: dict) -> TaskStepNode:
    analysis_type = analysis_dict[TYPE_KEY]
    data_dict = analysis_dict[DATA_KEY]
    doc: TaskStepNode
 
    if analysis_type == TaskStepNode.get_type():
        doc = TaskStepNode.model_validate(data_dict)
    else:
        raise ValueError(f"Unknown doc type: {analysis_type}")

    return doc

 
