from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import Field

from dreamsboard.engine.schema import BaseNode, ObjectTemplateType
from dreamsboard.templates import get_template_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


class DreamsPersonalityNode(BaseNode, ABC):
    """心里分析信息节点"""

    story_scenario_context: Optional[str] = Field(
        default="", description="开放问题 故事情境"
    )
    scene_monologue_context: Optional[str] = Field(
        default="", description="开放问题 故事场景"
    )
    user_id: Optional[str] = Field(default="", description="开放问题user_id")
    scene_content: Optional[str] = Field(default="", description="开放问题 文本内容")
    story_board_summary_context: Optional[str] = Field(
        default="", description="开放问题 人物对话"
    )
    dreams_guidance_context: Optional[str] = Field(
        default="", description="开放性问题dreams_guidance_context"
    )
    evolutionary_step: Optional[str] = Field(
        default="",
        description="性格信息 剧情总结",
    )
    dreams_personality_context: Optional[str] = Field(
        default="",
        description="性格信息dreams_personality_context",
    )
    ref_analysis_id: Optional[str] = Field(
        default="",
        description="ref_analysis_id",
    )

    def __init__(
        self,
        story_scenario_context: str = None,
        scene_monologue_context: str = None,
        user_id: str = None,
        scene_content: str = None,
        story_board_summary_context: str = None,
        dreams_guidance_context: str = None,
        evolutionary_step: str = None,
        dreams_personality_context=None,
        ref_analysis_id: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.story_scenario_context = story_scenario_context
        self.scene_monologue_context = scene_monologue_context
        self.user_id = user_id
        self.scene_content = scene_content
        self.story_board_summary_context = story_board_summary_context
        self.evolutionary_step = evolutionary_step
        self.dreams_guidance_context = dreams_guidance_context
        self.dreams_personality_context = dreams_personality_context
        self.ref_analysis_id = ref_analysis_id

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        story_scenario_context = cfg.get("story_scenario_context", "")
        scene_monologue_context = cfg.get("scene_monologue_context", "")
        user_id = cfg.get("user_id", "")
        scene_content = cfg.get("scene_content", "")
        story_board_summary_context = cfg.get("story_board_summary_context", "")
        evolutionary_step = cfg.get("evolutionary_step", "")
        dreams_guidance_context = cfg.get("dreams_guidance_context", "")
        dreams_personality_context = cfg.get("dreams_personality_context", "")
        ref_analysis_id = cfg.get("ref_analysis_id", "")
        return cls(
            story_scenario_context=story_scenario_context,
            scene_monologue_context=scene_monologue_context,
            user_id=user_id,
            scene_content=scene_content,
            story_board_summary_context=story_board_summary_context,
            evolutionary_step=evolutionary_step,
            dreams_guidance_context=dreams_guidance_context,
            dreams_personality_context=dreams_personality_context,
            ref_analysis_id=ref_analysis_id,
        )

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return "dreams_node"

    def class_name(self) -> str:
        """Get class name."""
        if hasattr(self, "__class_getitem__"):
            return self.__class__.__name__
        elif hasattr(self, "__orig_class__"):
            return self.__orig_class__.__name__
        elif hasattr(self, "__name__"):
            return self.__name__
        else:
            return "心里分析信息节点"

    @property
    def template_content(self) -> str:
        return (
            f"story_scenario_context: {self.story_scenario_context}, "
            f"scene_monologue_context: {self.scene_monologue_context}, "
            f"user_id: {self.user_id}, "
            f"scene_content: {self.scene_content}, "
            f"story_board_summary_context: {self.story_board_summary_context}, "
            f"dreams_guidance_context: {self.dreams_guidance_context}, "
            f"evolutionary_step: {self.evolutionary_step}, "
            f"dreams_personality_context: {self.dreams_personality_context}, "
            f"ref_analysis_id: {self.ref_analysis_id}"
        )

    @template_content.setter
    def template_content(self, _template_content) -> None:
        raise RuntimeError("template_content is read only.")
