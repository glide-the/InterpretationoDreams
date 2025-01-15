
TASK_MD_TEMPLATE = """
# {start_task_context} 


{context_placeholder}
 
"""



TASK_STEP_MD_TITLE_TEMPLATE = """
{task_step_name} [task_id:{task_step_level}]({task_step_id})

{task_step_description}
"""


TASK_STEP_MD_DESC_TEMPLATE = """
{task_step_name} [task_id:{task_step_level}]({task_step_id}){task_step_description} {task_step_question_answer}
"""



TASK_STEP_MD_LIST_TEMPLATE = """
- **{task_step_name} [task_id:{task_step_level}]({task_step_id})**:{task_step_question_answer}
"""


TASK_STEP_MD_TEMPLATE = """
{task_step_name} [task_id:{task_step_level}]({task_step_id}), {task_step_question_answer}
"""
