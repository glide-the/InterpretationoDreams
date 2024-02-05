from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Any, Optional, Dict

import logging

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
    dreams_guidance_context: Optional[str] = Field(
        default="", description="开放问题"
    )
    dreams_personality_context: Optional[str] = Field(
        default="", description="性格信息",
    )

    def __init__(self, dreams_guidance_context: str = None, dreams_personality_context=None, **kwargs):
        super().__init__(**kwargs)
        self.dreams_guidance_context = dreams_guidance_context
        self.dreams_personality_context = dreams_personality_context

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        dreams_guidance_context = cfg.get("dreams_guidance_context", "")
        dreams_personality_context = cfg.get("dreams_personality_context", "")
        return cls(dreams_guidance_context=dreams_guidance_context,
                   dreams_personality_context=dreams_personality_context)

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return "dreams_node"

    def class_name(self) -> str:
        """Get class name."""
        return self.__name__

    @property
    def template_content(self) -> str:
        return "dreams_personality: " + self.dreams_personality + "\r\ndreams_guidance_context: " + self.dreams_guidance_context

    @template_content.setter
    def template_content(self, _template_content) -> None:
        raise RuntimeError("template_content is read only.")
