from langchain_core.pydantic_v1 import BaseModel, Field


class Personality(BaseModel):
    """性格信息."""

    personality: str = Field(..., description="性格评价")


class DreamsStepInfo(BaseModel):
    """开放性引导问题"""

    step_advice: str = Field(
        ...,
        description="""Advice provided in this step, e.g. "I would say something like: 'I understand this is a difficult situation for you.'""",
    )
    step_description: str = Field(
        ...,
        description="""(Description of the counseling step, e.g. "Establish trust" """,
    )


class DreamsStepInfoListWrapper(BaseModel):
    """开放问题询问步骤列表."""

    steps: list[DreamsStepInfo] = Field(
        ..., description="""List of steps in the counseling process"""
    )


class TaskStepNode(BaseModel):
    """任务步骤节点."""

    start_task_context: str = Field(..., description="""开始任务""")
    aemo_representation_context: str = Field(..., description=""" 任务总体描述""")
    task_step_name: str = Field(
        ...,
        description="""提取步骤的名称，例如"分析近几年研究领域的技术框架与方法论"、"基于规则的方法"、"研究论文中采用的主要框架"、"Transformer"等""",
    )
    task_step_description: str = Field(
        ...,
        description="""提取每个步骤的具体建议和问题，例如"Text2SQL 是将自然语言查询（NLQ）转换为结构化查询语言（SQL）的任务，近年来在数据库和自然语言处理（NLP）领域受到广泛关注。主要技术框架和方法论包括："等""",
    )
    task_step_level: str = Field(
        ...,
        description="""提取步骤的层级编号，只有root>child层级，例如"0"、"0>1"、"0>2"、"1"、"1>1"、"1>2"等""",
    )


class TaskStepNodeListWrapper(BaseModel):
    """任务步骤列表."""

    steps: list[TaskStepNode] = Field(
        ..., description="""List of steps in the counseling process"""
    )

    def to_dict(self):
        return [step.to_dict() for step in self.steps]


class TaskStepRefineNode(BaseModel):
    """根据批评意见优化当前回答并续写上下文内容."""

    thought: str = Field(..., description="""优化后的回答""")
    answer: str = Field(..., description=""" 续写的上下文内容""")
    answer_socre: str = Field(..., description="""答案评分分数。 浮点数""")
