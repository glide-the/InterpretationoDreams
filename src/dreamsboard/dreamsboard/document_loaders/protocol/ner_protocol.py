from langchain_core.pydantic_v1 import BaseModel, Field


class Personality(BaseModel):
    """性格信息."""

    personality: str = Field(..., description="性格评价")


class DreamsStepInfo(BaseModel):
    """开放性引导问题"""
    step_advice: str = Field(..., description="""Advice provided in this step, e.g. "I would say something like: 'I understand this is a difficult situation for you.'""")
    step_description: str = Field(..., description="""(Description of the counseling step, e.g. "Establish trust" """)


class DreamsStepInfoListWrapper(BaseModel):
    """开放问题询问步骤列表."""
    steps: list[DreamsStepInfo] = Field(..., description="""List of steps in the counseling process""")
