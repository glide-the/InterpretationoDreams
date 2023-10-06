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


# 创建一个代码生成器抽象类
class CodeGenerator(BaseNode, ABC):

    @classmethod
    def from_config(cls, cfg=None):
        return cls()


# 创建不同类型的代码生成器
class BaseProgramGenerator(CodeGenerator):
    exec_code: Optional[str] = Field(
        default="", description="执行代码"
    )
    base_template_content: Optional[str] = Field(
        default="", description="模板内容",
    )
    exec_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="执行环境数据",
        alias="exec_data",
    )

    def __init__(self, code_file: str = None, render_data=None, **kwargs):
        # 检查kwargs中是否包含这些属性，如果包含就不执行文件读取
        if "exec_data" not in kwargs and "base_template_content" not in kwargs and "exec_code" not in kwargs:
            if render_data is None:
                render_data = {}
            # 读取模板文件
            with open(code_file, 'r') as template_file:
                base_template_content = template_file.read()

            super().__init__(exec_data=render_data, base_template_content=base_template_content)
        else:
            super().__init__(**kwargs)

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectTemplateType.BaseProgramGenerator

    def class_name(self) -> str:
        """Get class name."""
        return self.__name__

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
    exec_code: Optional[str] = Field(
        default="", description="执行代码"
    )
    dreams_query_template_content: Optional[str] = Field(
        default="", description="模板内容",
    )
    exec_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="执行环境数据",
        alias="exec_data",
    )

    def __init__(self, dreams_query_code_file: str = None, render_data=None, **kwargs):

        # 检查kwargs中是否包含这些属性，如果包含就不执行文件读取
        if "exec_data" not in kwargs and "base_template_content" not in kwargs and "exec_code" not in kwargs:
            if render_data is None:
                render_data = {}
            # 读取模板文件
            with open(dreams_query_code_file, 'r') as dreams_query_template_file:
                dreams_query_template_content = dreams_query_template_file.read()

            self.dreams_query_template_content = dreams_query_template_content

            super().__init__(exec_data=render_data, dreams_query_template_content=dreams_query_template_content)
        else:
            super().__init__(**kwargs)

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectTemplateType.QueryProgramGenerator

    def class_name(self) -> str:
        """Get class name."""
        return self.__name__

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        dreams_query_code_file = cfg.get("dreams_query_code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(dreams_query_code_file=get_template_path(dreams_query_code_file), render_data=render_data)

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
    exec_code: Optional[str] = Field(
        default="", description="执行代码"
    )
    ai_template_content: Optional[str] = Field(
        default="", description="模板内容",
    )
    exec_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="执行环境数据",
        alias="exec_data",
    )

    def __init__(self, ai_code_file: str = None, render_data=None, **kwargs):

        # 检查kwargs中是否包含这些属性，如果包含就不执行文件读取
        if "exec_data" not in kwargs and "base_template_content" not in kwargs and "exec_code" not in kwargs:
            if render_data is None:
                render_data = {}
            # 读取模板文件
            with open(ai_code_file, 'r') as ai_template_file:
                ai_template_content = ai_template_file.read()

            super().__init__(exec_data=render_data, ai_template_content=ai_template_content)
        else:
            super().__init__(**kwargs)

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectTemplateType.AIProgramGenerator

    def class_name(self) -> str:
        """Get class name."""
        return self.__name__

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        ai_code_file = cfg.get("ai_code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(ai_code_file=get_template_path(ai_code_file), render_data=render_data)

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
    exec_code: Optional[str] = Field(
        default="", description="执行代码"
    )
    engine_template_content: Optional[str] = Field(
        default="", description="模板内容",
    )
    exec_data: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="执行环境数据",
        alias="exec_data",
    )

    def __init__(self, engine_code_file: str = None, render_data=None, **kwargs):

        # 检查kwargs中是否包含这些属性，如果包含就不执行文件读取
        if "exec_data" not in kwargs and "base_template_content" not in kwargs and "exec_code" not in kwargs:
            if render_data is None:
                render_data = {}
            # 读取模板文件
            with open(engine_code_file, 'r') as engine_template_file:
                engine_template_content = engine_template_file.read()

            super().__init__(exec_data=render_data, engine_template_content=engine_template_content)
        else:
            super().__init__(**kwargs)

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectTemplateType.EngineProgramGenerator

    def class_name(self) -> str:
        """Get class name."""
        return self.__name__

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        engine_code_file = cfg.get("engine_code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(engine_code_file=get_template_path(engine_code_file), render_data=render_data)

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
