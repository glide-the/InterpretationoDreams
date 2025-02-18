from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from dreamsboard.engine.schema import BaseNode, ObjectTemplateType
from dreamsboard.templates import get_template_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


class TaskStepContext(BaseModel):
    ref_id: str = Field(default="", description="ref_id")
    paper_title: str = Field(default="", description="paper_title")
    chunk_id: str = Field(default="", description="chunk_id")
    score: float = Field(default=0.0, description="score")
    text: str = Field(default="", description="text")


class TaskStepNode(BaseNode, ABC):
    """任务步骤节点"""

    start_task_context: Optional[str] = Field(default="", description="开始任务")
    aemo_representation_context: Optional[str] = Field(
        default="", description="任务总体描述"
    )
    task_step_name: Optional[str] = Field(default="", description="任务步骤名称")
    task_step_description: Optional[str] = Field(default="", description="任务步骤描述")
    task_step_level: Optional[str] = Field(default="", description="任务步骤层级")
    task_step_question: Optional[str] = Field(default="", description="任务步骤问题")
    task_step_question_context: Optional[List[TaskStepContext]] = Field(
        default=[], description="任务步骤问题上下文"
    )

    task_step_question_answer: Optional[str] = Field(
        default="", description="任务步骤答案"
    )
    ref_task_step_id: Optional[str] = Field(
        default="",
        description="ref_task_step_id",
    )

    def __init__(
        self,
        start_task_context: str = None,
        aemo_representation_context: str = None,
        task_step_name: str = None,
        task_step_description: str = None,
        task_step_level: str = None,
        task_step_question: str = None,
        task_step_question_context: List[TaskStepContext] = None,
        task_step_question_answer: str = None,
        ref_task_step_id: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.start_task_context = start_task_context
        self.aemo_representation_context = aemo_representation_context
        self.task_step_name = task_step_name
        self.task_step_description = task_step_description
        self.task_step_level = task_step_level
        self.task_step_question = task_step_question
        self.task_step_question_context = task_step_question_context
        self.task_step_question_answer = task_step_question_answer
        self.ref_task_step_id = ref_task_step_id

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        start_task_context = cfg.get("start_task_context", "")
        aemo_representation_context = cfg.get("aemo_representation_context", "")
        task_step_name = cfg.get("task_step_name", "")
        task_step_description = cfg.get("task_step_description", "")
        task_step_level = cfg.get("task_step_level", "")
        task_step_question = cfg.get("task_step_question", "")
        task_step_question_context = cfg.get("task_step_question_context", [])
        task_step_question_answer = cfg.get("task_step_question_answer", "")
        ref_task_step_id = cfg.get("ref_task_step_id", "")
        return cls(
            start_task_context=start_task_context,
            aemo_representation_context=aemo_representation_context,
            task_step_name=task_step_name,
            task_step_description=task_step_description,
            task_step_level=task_step_level,
            task_step_question=task_step_question,
            task_step_question_context=task_step_question_context,
            task_step_question_answer=task_step_question_answer,
            ref_task_step_id=ref_task_step_id,
        )

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return "task_step_node"

    def class_name(self) -> str:
        """Get class name."""
        if hasattr(self, "__class_getitem__"):
            return self.__class__.__name__
        elif hasattr(self, "__orig_class__"):
            return self.__orig_class__.__name__
        elif hasattr(self, "__name__"):
            return self.__name__
        else:
            return "任务步骤节点"

    @property
    def template_content(self) -> str:
        return (
            f"start_task_context: {self.start_task_context}, "
            f"aemo_representation_context: {self.aemo_representation_context}, "
            f"task_step_name: {self.task_step_name}, "
            f"task_step_description: {self.task_step_description}, "
            f"task_step_level: {self.task_step_level}, "
            f"task_step_question: {self.task_step_question}, "
            f"task_step_question_context: {self.task_step_question_context}, "
            f"task_step_question_answer: {self.task_step_question_answer}, "
            f"ref_task_step_id: {self.ref_task_step_id}"
        )

    @template_content.setter
    def template_content(self, _template_content) -> None:
        raise RuntimeError("template_content is read only.")
