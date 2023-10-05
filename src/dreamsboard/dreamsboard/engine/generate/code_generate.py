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
from typing import Any

import logging

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
    _exec_code: str
    _base_template_content: str
    _render_data: dict

    def __init__(self, code_file: str, render_data: dict = {}):
        super().__init__()
        self._exec_code = None
        self._render_data = render_data
        # 读取模板文件
        with open(code_file, 'r') as template_file:
            self._base_template_content = template_file.read()

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
        return self._base_template_content

    @template_content.setter
    def template_content(self, _template_content) -> None:
        self._base_template_content = _template_content

    @property
    def render_data(self) -> dict:
        return self._render_data

    @render_data.setter
    def render_data(self, _render_data: dict) -> None:
        self._render_data = _render_data

    @property
    def render_code(self) -> str:
        return self._exec_code

    @render_code.setter
    def render_code(self, _exec_code) -> None:
        self._exec_code = _exec_code


class QueryProgramGenerator(CodeGenerator):
    _exec_code: str
    _dreams_query_template_content: str
    _render_data: dict

    def __init__(self, dreams_query_code_file: str, render_data: dict = {}):
        super().__init__()
        self._exec_code = None
        self._render_data = render_data
        # 读取模板文件
        with open(dreams_query_code_file, 'r') as dreams_query_template_file:
            self._dreams_query_template_content = dreams_query_template_file.read()

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
        return self._dreams_query_template_content

    @template_content.setter
    def template_content(self, _template_content) -> None:
        self._dreams_query_template_content = _template_content

    @property
    def render_data(self) -> dict:
        return self._render_data

    @render_data.setter
    def render_data(self, _render_data: dict) -> None:
        self._render_data = _render_data

    @property
    def render_code(self) -> str:
        return self._exec_code

    @render_code.setter
    def render_code(self, _exec_code) -> None:
        self._exec_code = _exec_code


class AIProgramGenerator(CodeGenerator):
    _exec_code: str
    _ai_template_content: str
    _render_data: dict

    def __init__(self, ai_code_file: str, render_data: dict = {}):
        super().__init__()
        self._exec_code = None
        self._render_data = render_data
        # 读取模板文件
        with open(ai_code_file, 'r') as ai_template_file:
            self._ai_template_content = ai_template_file.read()

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
        return self._ai_template_content

    @template_content.setter
    def template_content(self, _template_content) -> None:
        self._ai_template_content = _template_content

    @property
    def render_data(self) -> dict:
        return self._render_data

    @render_data.setter
    def render_data(self, _render_data: dict) -> None:
        self._render_data = _render_data

    @property
    def render_code(self) -> str:
        return self._exec_code

    @render_code.setter
    def render_code(self, _exec_code) -> None:
        self._exec_code = _exec_code


class EngineProgramGenerator(CodeGenerator):
    _exec_code: str
    _engine_template_content: str
    _render_data: dict

    def __init__(self, engine_code_file: str, render_data: dict = {}):
        super().__init__()
        self._exec_code = None
        self._render_data = render_data
        # 读取模板文件
        with open(engine_code_file, 'r') as engine_template_file:
            self._engine_template_content = engine_template_file.read()

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
        return self._engine_template_content

    @template_content.setter
    def template_content(self, _template_content) -> None:
        self._engine_template_content = _template_content

    @property
    def render_data(self) -> dict:
        return self._render_data

    @render_data.setter
    def render_data(self, _render_data: dict) -> None:
        self._render_data = _render_data

    @property
    def render_code(self) -> str:
        return self._exec_code

    @render_code.setter
    def render_code(self, _exec_code) -> None:
        self._exec_code = _exec_code
