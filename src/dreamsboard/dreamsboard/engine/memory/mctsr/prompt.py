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
    critic_system_prompt="""# 执行任务：结合开始任务（start_task_context），在符合任务总体描述（aemo_representation_context）的情况下，根据任务步骤名称（task_step_name）、任务步骤描述（task_step_description）和任务步骤层级（task_step_level），
提供详细且有建设性的批评意见以改进`<current_answer>`。
突出显示需要改进或更正的特定区域。
<problem>
{problem}
</problem>
<current_answer>
{current_answer}
</current_answer>


### 补充指南  
- 不要重复`<problem>`描述。  
- 不要重复`<current_answer>`描述。
- 不要重复`<start_task_context>`描述。
- 不要重复`<aemo_representation_context>`描述。
- 不要重复`<task_step_name>`描述。
- 不要重复`<task_step_description>`描述。
- 不要重复`<task_step_level>`描述。


### 任务

start_task_context: {start_task_context}

aemo_representation_context: {aemo_representation_context}


### 当前任务信息

task_step_name: {task_step_name}
task_step_description: {task_step_description}
task_step_level: {task_step_level}


""",
    refine_system_prompt="""# 执行任务：结合开始任务（start_task_context），在符合任务总体描述（aemo_representation_context）的情况下，根据任务步骤名称（task_step_name）、任务步骤描述（task_step_description）和任务步骤层级（task_step_level），
根据批评意见(`<critique>`)优化`<current_answer>`。在思考过程中输出优化后的答案。

<problem>
{problem}
</problem>
<current_answer>
{current_answer}
</current_answer>
<critique>
{critique}
</critique>

### 补充指南  
- 不要重复`<critique>`描述。  
- 不要重复`<problem>`描述。   
- 不要重复`<start_task_context>`描述。
- 不要重复`<aemo_representation_context>`描述。
- 不要重复`<task_step_name>`描述。
- 不要重复`<task_step_description>`描述。
- 不要重复`<task_step_level>`描述。


### 任务

start_task_context: {start_task_context}

aemo_representation_context: {aemo_representation_context}


### 当前任务信息

task_step_name: {task_step_name}
task_step_description: {task_step_description}
task_step_level: {task_step_level}


# JSON 响应格式  
```json  
{{
    "thought": "对答案的思考过程。",
    "answer": "表示问题答案的浮点数。"
}}
```
""",
    evaluate_system_prompt="""# 执行任务：结合开始任务（start_task_context），在符合任务总体描述（aemo_representation_context）的情况下，根据任务步骤名称（task_step_name）、任务步骤描述（task_step_description）和任务步骤层级（task_step_level），
Provide a reward score between -100 and 100 for the answer quality, using very strict standards. 
Do not give a full score above 95. Make sure the reward score is an integer. 
Return *ONLY* the score. 

<problem>
{problem}
</problem>
<answer>
{answer}
</answer>


### 补充指南  
- 不要重复`<problem>`描述。  
- 不要重复`<current_answer>`描述。
- 不要重复`<start_task_context>`描述。
- 不要重复`<aemo_representation_context>`描述。
- 不要重复`<task_step_name>`描述。
- 不要重复`<task_step_description>`描述。
- 不要重复`<task_step_level>`描述。


### 任务

start_task_context: {start_task_context}

aemo_representation_context: {aemo_representation_context}


### 当前任务信息

task_step_name: {task_step_name}
task_step_description: {task_step_description}
task_step_level: {task_step_level}

""",
)
