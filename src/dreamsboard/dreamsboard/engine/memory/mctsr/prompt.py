from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    critic_system_prompt: str
    refine_system_prompt: str
    evaluate_system_prompt: str
 

class RefineResponse(BaseModel):
    thought: str = Field(..., description="The thought process behind the answer.")
    answer: float = Field(..., description="The answer to the problem.")


GLM_JSON_RESPONSE_PREFIX = """You should always follow the instructions and output a valid JSON object.
The structure of the JSON object you can found in the instructions, use {"answer": "$your_answer"} as the default structure
if you are not sure about the structure.

And you should always end the block with a "```" to indicate the end of the JSON object.

<instructions>
"""

GLM_JSON_RESPONSE_SUFFIX = """Output:
</instructions>

"""

gpt_prompt_config = PromptConfig(
    critic_system_prompt="""完成你的目标任务,输出详细且有建设性的批评意见以改进`<current_answer>`， step by step plan. 

你的目标:
<problem>
{problem}
</problem>

你目前的结果在这里:
<context>
{context}
</context>

<current_answer>
{current_answer}
</current_answer>

你目前已完成以下步骤：
{past_steps}


### 参考资源

start_task_context: {start_task_context}
aemo_representation_context: {aemo_representation_context}

### 当前任务信息

task_step_name: {task_step_name}
task_step_description: {task_step_description}
task_step_level: {task_step_level}

### 补充指南  

- 不要重复`<problem>`描述。  
- 不要重复`<current_answer>`描述。
- 不要重复`<start_task_context>`描述。
- 不要重复`<aemo_representation_context>`描述。
- 不要重复`<task_step_name>`描述。
- 不要重复`<task_step_description>`描述。
- 不要重复`<task_step_level>`描述。


结合开始任务（start_task_context），在符合任务总体描述（aemo_representation_context）的情况下，根据任务步骤名称（task_step_name）、任务步骤描述（task_step_description）和任务步骤层级（task_step_level），

突出显示需要改进或更正的特定区域。不需要更多步骤, 不要将之前完成的步骤作为计划的一部分返回。
""",
    refine_system_prompt="""完成你的目标任务,根据批评意见(`<critique>`)优化当前内容`<current_answer>`。输出优化答案和分数

你的目标:
<problem>
{problem}
</problem>

批评意见:
<critique>
{critique}
</critique>

你目前的结果在这里:
<context>
{context}
</context>

<current_answer>
{current_answer}
</current_answer>

你目前已完成以下步骤：
{past_steps}


### 补充指南  
- 不要重复`<critique>`描述。  
- 不要重复`<problem>`描述。   
- 不要重复`<start_task_context>`描述。
- 不要重复`<aemo_representation_context>`描述。
- 不要重复`<task_step_name>`描述。
- 不要重复`<task_step_description>`描述。
- 不要重复`<task_step_level>`描述。
 

### 响应格式包含`thought`、`answer`

thought: 输出的优化答案 
输出要求
    - 表格输出格式```plaintext ```
    - 其它文本不超过500字

answer: 答案评分分数。
输出要求
    - 浮点数
 
""",
    evaluate_system_prompt="""Provide a reward score between -100 and 100 for the answer quality, using very strict standards. 
Do not give a full score above 95. Make sure the reward score is an integer. 
Return *ONLY* the score. 

<problem>
{problem}
</problem>
<answer>
{answer}
</answer>
""",
)
