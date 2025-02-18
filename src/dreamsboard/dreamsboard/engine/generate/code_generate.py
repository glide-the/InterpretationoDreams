"""
使用设计模式，构建一个代码生成器，功能如下，
程序动态的生成python代码，python代码有三种形式，1、基础程序。2、逻辑控制程序。3、逻辑加载程序。4、逻辑运行程序

程序的执行流程如下：
1、
构建出的代码通过exec函数运行

2、
构建出的代码最终通过建造者生成一个执行器

3、
建造者在执行CodeGenerator的时候，需要使用责任链实现


"""

from __future__ import annotations

import hashlib
import logging
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from jinja2 import Template
from pydantic import Field

from dreamsboard.engine.schema import (
    TRUNCATE_LENGTH,
    WRAP_WIDTH,
    BaseNode,
    ObjectTemplateType,
)
from dreamsboard.templates import get_template_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


# 创建一个代码生成器抽象类
class CodeGenerator(BaseNode, ABC):
    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    @property
    @abstractmethod
    def render_data(self) -> dict:
        """Get render_data."""

    @render_data.setter
    @abstractmethod
    def render_data(self, _render_data: dict) -> None:
        """Get render_data."""

    @property
    @abstractmethod
    def render_code(self) -> str:
        """Get render_code."""

    @render_code.setter
    @abstractmethod
    def render_code(self, _exec_code: str) -> None:
        """Get render_code."""

    def generate(self, render_data: dict = {}) -> str:
        base_template = Template(self.template_content)
        if render_data is not None and self.render_data is not None:
            self.render_data = {**render_data, **self.render_data}
        else:
            # 处理其中一个或两者都为 None 的情况
            self.render_data = render_data or self.render_data or {}

        self.render_code = base_template.render(self.render_data)

        return self.render_code

    def calculate_md5(self):
        md5_hash = hashlib.md5()
        md5_hash.update(self.render_code.encode("utf-8"))
        return md5_hash.hexdigest()

    def __str__(self) -> str:
        source_text_truncated = truncate_text(self.render_code.strip(), TRUNCATE_LENGTH)
        source_text_wrapped = textwrap.fill(
            f"Text: {source_text_truncated}\n", width=WRAP_WIDTH
        )
        return f"Node ID: {self.node_id}\n{source_text_wrapped}"


# 创建不同类型的代码生成器
class BaseProgramGenerator(CodeGenerator):
    exec_code: Optional[str] = Field(default="", description="执行代码")
    base_template_content: Optional[str] = Field(
        default="",
        description="模板内容",
    )
    exec_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="执行环境数据",
        alias="exec_data",
    )

    def __init__(self, code_file: str = None, render_data=None, **kwargs):
        # 检查kwargs中是否包含这些属性，如果包含就不执行文件读取
        if (
            "exec_data" not in kwargs
            and "base_template_content" not in kwargs
            and "exec_code" not in kwargs
        ):
            if render_data is None:
                render_data = {}
            # 读取模板文件
            with open(code_file, "r", encoding="utf-8") as template_file:
                base_template_content = template_file.read()

            super().__init__(
                exec_data=render_data, base_template_content=base_template_content
            )
        else:
            super().__init__(**kwargs)

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectTemplateType.BaseProgramGenerator

    def class_name(self) -> str:
        """Get class name."""
        if hasattr(self, "__class_getitem__"):
            return self.__class__.__name__
        elif hasattr(self, "__orig_class__"):
            return self.__orig_class__.__name__
        elif hasattr(self, "__name__"):
            return self.__name__
        else:
            raise RuntimeError("class_name is None.")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        code_file = cfg.get("code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(code_file=get_template_path(code_file), render_data=render_data)

    @property
    def template_content(self) -> str:
        return self.base_template_content

    @template_content.setter
    def template_content(self, _template_content) -> None:
        self.base_template_content = _template_content

    @property
    def render_data(self) -> dict:
        return self.exec_data

    @render_data.setter
    def render_data(self, exec_data: dict) -> None:
        self.exec_data = exec_data

    @property
    def render_code(self) -> str:
        return self.exec_code

    @render_code.setter
    def render_code(self, exec_code) -> None:
        self.exec_code = exec_code


class QueryProgramGenerator(CodeGenerator):
    exec_code: Optional[str] = Field(default="", description="执行代码")
    dreams_query_template_content: Optional[str] = Field(
        default="",
        description="模板内容",
    )
    exec_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="执行环境数据",
        alias="exec_data",
    )

    def __init__(self, query_code_file: str = None, render_data=None, **kwargs):
        # 检查kwargs中是否包含这些属性，如果包含就不执行文件读取
        if (
            "exec_data" not in kwargs
            and "base_template_content" not in kwargs
            and "exec_code" not in kwargs
        ):
            if render_data is None:
                render_data = {}
            # 读取模板文件
            with open(
                query_code_file, "r", encoding="utf-8"
            ) as dreams_query_template_file:
                dreams_query_template_content = dreams_query_template_file.read()

            super().__init__(
                exec_data=render_data,
                dreams_query_template_content=dreams_query_template_content,
            )
        else:
            super().__init__(**kwargs)

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectTemplateType.QueryProgramGenerator

    def class_name(self) -> str:
        """Get class name."""
        if hasattr(self, "__class_getitem__"):
            return self.__class__.__name__
        elif hasattr(self, "__orig_class__"):
            return self.__orig_class__.__name__
        elif hasattr(self, "__name__"):
            return self.__name__
        else:
            raise RuntimeError("class_name is None.")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        query_code_file = cfg.get("query_code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(
            query_code_file=get_template_path(query_code_file), render_data=render_data
        )

    @property
    def template_content(self) -> str:
        return self.dreams_query_template_content

    @template_content.setter
    def template_content(self, _template_content) -> None:
        self.dreams_query_template_content = _template_content

    @property
    def render_data(self) -> dict:
        return self.exec_data

    @render_data.setter
    def render_data(self, exec_data: dict) -> None:
        self.exec_data = exec_data

    @property
    def render_code(self) -> str:
        return self.exec_code

    @render_code.setter
    def render_code(self, exec_code) -> None:
        self.exec_code = exec_code


class AIProgramGenerator(CodeGenerator):
    exec_code: Optional[str] = Field(default="", description="执行代码")
    ai_template_content: Optional[str] = Field(
        default="",
        description="模板内容",
    )
    exec_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="执行环境数据",
        alias="exec_data",
    )

    def __init__(self, ai_code_file: str = None, render_data=None, **kwargs):
        # 检查kwargs中是否包含这些属性，如果包含就不执行文件读取
        if (
            "exec_data" not in kwargs
            and "base_template_content" not in kwargs
            and "exec_code" not in kwargs
        ):
            if render_data is None:
                render_data = {}
            # 读取模板文件
            with open(ai_code_file, "r", encoding="utf-8") as ai_template_file:
                ai_template_content = ai_template_file.read()

            super().__init__(
                exec_data=render_data, ai_template_content=ai_template_content
            )
        else:
            super().__init__(**kwargs)

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectTemplateType.AIProgramGenerator

    def class_name(self) -> str:
        """Get class name."""
        if hasattr(self, "__class_getitem__"):
            return self.__class__.__name__
        elif hasattr(self, "__orig_class__"):
            return self.__orig_class__.__name__
        elif hasattr(self, "__name__"):
            return self.__name__
        else:
            raise RuntimeError("class_name is None.")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        ai_code_file = cfg.get("ai_code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(
            ai_code_file=get_template_path(ai_code_file), render_data=render_data
        )

    @property
    def template_content(self) -> str:
        return self.ai_template_content

    @template_content.setter
    def template_content(self, _template_content) -> None:
        self.ai_template_content = _template_content

    @property
    def render_data(self) -> dict:
        return self.exec_data

    @render_data.setter
    def render_data(self, exec_data: dict) -> None:
        self.exec_data = exec_data

    @property
    def render_code(self) -> str:
        return self.exec_code

    @render_code.setter
    def render_code(self, exec_code) -> None:
        self.exec_code = exec_code


class EngineProgramGenerator(CodeGenerator):
    exec_code: Optional[str] = Field(default="", description="执行代码")
    engine_template_content: Optional[str] = Field(
        default="",
        description="模板内容",
    )
    exec_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="执行环境数据",
        alias="exec_data",
    )

    def __init__(self, engine_code_file: str = None, render_data=None, **kwargs):
        # 检查kwargs中是否包含这些属性，如果包含就不执行文件读取
        if (
            "exec_data" not in kwargs
            and "base_template_content" not in kwargs
            and "exec_code" not in kwargs
        ):
            if render_data is None:
                render_data = {
                    "model_name": "gpt-4",
                }
            model_name = render_data.get("model_name", None)
            if model_name is None:
                raise RuntimeError("model_name is None.")
            # 读取模板文件
            with open(engine_code_file, "r", encoding="utf-8") as engine_template_file:
                engine_template_content = engine_template_file.read()

            super().__init__(
                exec_data=render_data, engine_template_content=engine_template_content
            )
        else:
            super().__init__(**kwargs)

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectTemplateType.EngineProgramGenerator

    def class_name(self) -> str:
        """Get class name."""
        if hasattr(self, "__class_getitem__"):
            return self.__class__.__name__
        elif hasattr(self, "__orig_class__"):
            return self.__orig_class__.__name__
        elif hasattr(self, "__name__"):
            return self.__name__
        else:
            raise RuntimeError("class_name is None.")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        engine_code_file = cfg.get("engine_code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(
            engine_code_file=get_template_path(engine_code_file),
            render_data=render_data,
        )

    @property
    def template_content(self) -> str:
        return self.engine_template_content

    @template_content.setter
    def template_content(self, _template_content) -> None:
        self.engine_template_content = _template_content

    @property
    def render_data(self) -> dict:
        return self.exec_data

    @render_data.setter
    def render_data(self, exec_data: dict) -> None:
        self.exec_data = exec_data

    @property
    def render_code(self) -> str:
        return self.exec_code

    @render_code.setter
    def render_code(self, exec_code) -> None:
        self.exec_code = exec_code
