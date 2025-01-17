import logging

import langchain

from dreamsboard.dreams.task_step_md.prompts import TASK_MD_TEMPLATE,TASK_STEP_MD_TITLE_TEMPLATE,TASK_STEP_MD_DESC_TEMPLATE,TASK_STEP_MD_LIST_TEMPLATE, TASK_STEP_MD_TEMPLATE
from dreamsboard.engine.storage.task_step_store.types import BaseTaskStepStore
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompt_values import StringPromptValue

from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_1 = PromptTemplate(input_variables=[
    "task_step_name",
    "task_step_level",
    "task_step_id",
    "task_step_question_answer"],
    template=TASK_STEP_MD_TITLE_TEMPLATE)
_PROMPT_TEMPLATE_1_1 = PromptTemplate(input_variables=[
    "task_step_name",
    "task_step_level",
    "task_step_id",
    "task_step_question_answer"],
    template=TASK_STEP_MD_DESC_TEMPLATE)
_PROMPT_TEMPLATE_1_2 = PromptTemplate(input_variables=[
    "task_step_name",
    "task_step_level",
    "task_step_id",
    "task_step_question_answer"],
    template=TASK_STEP_MD_LIST_TEMPLATE)
_PROMPT_TEMPLATE_1_3 = PromptTemplate(input_variables=[
    "task_step_name",
    "task_step_level",
    "task_step_id",
    "task_step_question_answer"],
    template=TASK_STEP_MD_TEMPLATE)
_PROMPT_TEMPLATE_2 = PromptTemplate(
    input_variables=[
        "start_task_context",
        "aemo_representation_context",
        "context_placeholder"
    ],
    template=TASK_MD_TEMPLATE)


class TaskStepMD:

    def __init__(self,
                 task_step_store: BaseTaskStepStore
                 ):
        self.task_step_store = task_step_store

    def format_md(self) -> StringPromptValue:
 
        
        def wrapper_steps_unit(dict_input: dict):
            # 使用 TASK_STEP_MD_TEMPLATE 格式化每个任务步骤
            formatted_task_steps = [
            ]
            for step in list(self.task_step_store.task_step_all.values()):
               # 计算层级关系
                level_count = step.task_step_level.count('>')

                if level_count == 0:
                    # 一级，格式化为标题 #
                    step_text = _PROMPT_TEMPLATE_1.format(
                        task_step_name=f'### {step.task_step_name}',
                        task_step_level=step.task_step_level,
                        task_step_description=step.task_step_description,
                        task_step_id=step.node_id,
                        task_step_question_answer=step.task_step_question_answer
                    )
                    formatted_task_steps.append(step_text.strip()+"\n\n")

                elif level_count == 1:
                    # 二级，格式化为标题 ##
                    step_text = _PROMPT_TEMPLATE_1_1.format(
                        task_step_name=f'{step.task_step_name}',
                        task_step_level=step.task_step_level,
                        task_step_description=step.task_step_description,
                        task_step_id=step.node_id,
                        task_step_question_answer=step.task_step_question_answer
                    )
                    formatted_task_steps.append(step_text.strip()+"\n\n")

                elif level_count >= 2:
                    # 三级及以上，格式化为分类 -
                    step_text = _PROMPT_TEMPLATE_1_2.format(
                        task_step_name=f'- {step.task_step_name}',
                        task_step_level=step.task_step_level,
                        task_step_description=step.task_step_description,
                        task_step_id=step.node_id,
                        task_step_question_answer=step.task_step_question_answer
                    )
                    formatted_task_steps.append(step_text.strip()+"\n\n")

                else:
                    step_text = _PROMPT_TEMPLATE_1_3.format(
                        task_step_name=f'{step.task_step_name}',
                        task_step_level=step.task_step_level,
                        task_step_description=step.task_step_description,
                        task_step_id=step.node_id,
                        task_step_question_answer=step.task_step_question_answer
                    )
                
                    formatted_task_steps.append(step_text.strip())
                

            # 将格式化的步骤列表转换为字符串
            context_placeholder = "".join(formatted_task_steps)
            return {
                "start_task_context": list(self.task_step_store.task_step_all.values())[0].start_task_context,
                "aemo_representation_context": list(self.task_step_store.task_step_all.values())[0].aemo_representation_context,
                "context_placeholder": context_placeholder
            }
 

        chain = (RunnableLambda(wrapper_steps_unit)  
                 | _PROMPT_TEMPLATE_2)

        out = chain.invoke({})
        return out

    def write_md(self, output_path: str) -> StringPromptValue:
        md = self.format_md()
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(md.text)
        logger.info(f"Write MD to {output_path}")
        return md
