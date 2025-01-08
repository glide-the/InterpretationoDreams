"""

"""

# 子任务指令转换为子问题.txt
CONVERT_TASK_STEP_TO_QUESTION_PROMPT_TEMPLATE = """执行任务：结合开始任务（start_task_context），在符合任务总体描述（aemo_representation_context）的情况下，根据任务步骤名称（task_step_name）、任务步骤描述（task_step_description）和任务步骤层级（task_step_level），提出一个问题

### 任务

start_task_context: {start_task_context}

aemo_representation_context: {aemo_representation_context}


### 当前任务信息

task_step_name: {task_step_name}
task_step_description: {task_step_description}
task_step_level: {task_step_level}
"""
 