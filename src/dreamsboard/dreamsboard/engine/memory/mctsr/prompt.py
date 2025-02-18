from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    critic_system_prompt: str
    critic_system_prompt_data: str
    refine_system_prompt: str
    refine_system_prompt_data: str
    evaluate_system_prompt: str
    evaluate_system_prompt_data: str


class RefineResponse(BaseModel):
    thought: str = Field(..., description="The thought process behind the answer.")
    answer: str = Field(..., description="The thought process behind the answer.")
    answer_score: float = Field(..., description="The answer to the problem.")


gpt_prompt_config = PromptConfig(
    critic_system_prompt="""完成你的目标任务,输出详细且有建设性的批评意见以改进`<current_answer>`， step by step plan. 

# 补充指南  

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
    critic_system_prompt_data="""你的目标:
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


# 参考资源

start_task_context: {start_task_context}
aemo_representation_context: {aemo_representation_context}

# 当前任务信息

task_step_name: {task_step_name}
task_step_description: {task_step_description}
task_step_level: {task_step_level}
""",
    refine_system_prompt="""作为一个计算机科学领域的研究者，您已经查阅了近年来在顶级会议（如 NeurIPS、CVPR、ICML）及顶级期刊（如 JMLR、IEEE TPAMI、ACM Computing Surveys）上的相关文献，并且深入分析了 arXiv 上的最新论文，尝试通过参考文献中定义的算法和模型构建， 完成你的目标任务
# 目标
根据批评意见 (`<critique>`) 优化当前回答 (`<current_answer>`) 并续写上下文 （`<context>`）。

### 补充指南  

- 不要重复`<problem>`描述。  
- 不要重复`<current_answer>`描述。 
- 结合总体任务进度输出 
- 不要做总结的回复

# 输出优化后的回答

### 优化后的回答: 输出的优化答案 
输出要求 
    - Does not want answers with numbered or structured content like 1. 2. 3. 4. 

### 续写的上下文: 包含续写的上下文
输出要求
    - 不允许出现`标题`、`分段`、`综上所述`的文字

### 评分分数：包含续写的评分分数
输出要求
    - 格式`分数/总分`

""",
    refine_system_prompt_data="""# 当前上下文：
<context>
{context}
</context>

# 当前答案：
<current_answer>
{current_answer}
</current_answer>

# 问题描述：
<problem>
{problem}
</problem>

# 批评意见：
<critique>
{critique}
</critique>

# 已完成的步骤：
{past_steps}
""",
    evaluate_system_prompt="""# Provide a reward score between -100 and 100 for the answer quality, using very strict standards. 
- Do not give a full score above 95. Ensure the reward score is an integer.
- Does not want answers with numbered or structured content like 1. 2. 3. 4. 
- Avoid using any segmented or explicitly structured terms. If disallowed content appears, assign a score of 10. 
- Return *ONLY* the score. 
- 只返回数字，不要返回其它内容
""",
    evaluate_system_prompt_data="""你的目标:
<problem>
{problem}
</problem>

你目前的结果在这里:
<context>
{context}
</context>

<answer>
{answer}
</answer>

你目前已完成以下步骤：
{past_steps} 

""",
)
