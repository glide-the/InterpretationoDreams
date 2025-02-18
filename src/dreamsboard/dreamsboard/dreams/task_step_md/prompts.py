TASK_MD_TEMPLATE = """
# {start_task_context} 


{context_placeholder}


# References  

{references}
"""


TASK_STEP_MD_TITLE_TEMPLATE = """
{task_step_name} [task_id]({task_step_id})<sup>{task_step_level}</sup>

{task_step_question_answer}
"""


TASK_STEP_MD_DESC_TEMPLATE = """
{task_step_name} [task_id]({task_step_id})<sup>{task_step_level}</sup> {task_step_question_answer}
"""


TASK_STEP_MD_LIST_TEMPLATE = """
- **{task_step_name} [task_id]({task_step_id})<sup>{task_step_level}</sup>**:{task_step_question_answer}
"""


TASK_STEP_MD_TEMPLATE = """
{task_step_name} [task_id]({task_step_id})<sup>{task_step_level}</sup>, {task_step_question_answer}
"""


TASK_REF_TEMPLATE = """[{task_step_level}] {paper_title} ,chunk_id:{ref_id} 
"""
