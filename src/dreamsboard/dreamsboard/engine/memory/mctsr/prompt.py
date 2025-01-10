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
    critic_system_prompt="Provide a detailed and constructive critique to improve the answer. "
    "Highlight specific areas that need refinement or correction.",
    refine_system_prompt="""根据批评意见优化答案。优化后的答案应直接且简洁地解决问题。

## 补充指南  
- 不应提及或讨论批评内容。  
- 不要重复问题描述。  

# JSON 响应格式  
```json  
{
    "thought": "对答案的思考过程。",
    "answer": "表示问题答案的浮点数。"
}
""",
    evaluate_system_prompt="Provide a reward score between -100 and 100 for the answer quality, using very strict standards. "
    "Do not give a full score above 95. Make sure the reward score is an integer. "
    "Return *ONLY* the score.",
)
