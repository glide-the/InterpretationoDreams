from dreamsboard.engine.constants import DATA_KEY, TYPE_KEY
from dreamsboard.engine.entity.task_step.task_step import TaskStepNode
from dreamsboard.engine.schema import (
    BaseNode,
)


def task_step_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_task_step(task_step_dict: dict) -> TaskStepNode:
    task_step_type = task_step_dict[TYPE_KEY]
    data_dict = task_step_dict[DATA_KEY]
    doc: TaskStepNode
 
    if task_step_type == TaskStepNode.get_type():
        doc = TaskStepNode.model_validate(data_dict)
    else:
        raise ValueError(f"Unknown doc type: {task_step_type}")

    return doc

 
