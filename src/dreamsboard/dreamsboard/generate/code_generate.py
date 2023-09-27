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
from jinja2 import Template
import hashlib
import logging

from dreamsboard.templates import get_template_path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)


# 创建一个代码生成器接口
class CodeGenerator:
    def __init__(self):
        self.transform = lambda x: x
        return

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    @property
    def render_code(self) -> str:
        raise NotImplementedError

    def generate(self, render_data: dict = {}) -> str:
        raise NotImplementedError

    def calculate_md5(self):
        md5_hash = hashlib.md5()
        md5_hash.update(self.render_code.encode('utf-8'))
        return md5_hash.hexdigest()


# 创建不同类型的代码生成器
class BaseProgramGenerator(CodeGenerator):

    _exec_code: str

    def __init__(self, code_file: str, render_data: dict = {}):
        super().__init__()
        self.render_data = render_data
        # 读取模板文件
        with open(code_file, 'r') as template_file:
            self.base_template_content = template_file.read()

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        code_file = cfg.get("code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(code_file=get_template_path(code_file), render_data=render_data)

    @property
    def render_code(self) -> str:
        return self._exec_code

    def generate(self, render_data: dict = {}) -> str:
        logger.info(f'{self.__class__},生成代码')
        base_template = Template(self.base_template_content)
        if render_data is not None and self.render_data is not None:
            merged_render = {**render_data, **self.render_data}
        else:
            # 处理其中一个或两者都为 None 的情况
            merged_render = render_data or self.render_data or {}

        self._exec_code = base_template.render(merged_render)

        logger.info(f'{self.__class__},生成代码成功 {self.calculate_md5()}')
        return self._exec_code


class QueryProgramGenerator(CodeGenerator):

    _exec_code: str

    def __init__(self, dreams_query_code_file: str, render_data: dict = {}):
        super().__init__()
        self.render_data = render_data
        # 读取模板文件
        with open(dreams_query_code_file, 'r') as dreams_query_template_file:
            self.dreams_query_template_content = dreams_query_template_file.read()

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        dreams_query_code_file = cfg.get("dreams_query_code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(dreams_query_code_file=get_template_path(dreams_query_code_file), render_data=render_data)

    @property
    def render_code(self) -> str:
        return self._exec_code

    def generate(self, render_data: dict = {}) -> str:
        logger.info(f'{self.__class__},生成代码')
        # 创建一个Jinja2模板对象
        dreams_query_template = Template(self.dreams_query_template_content)
        if render_data is not None and self.render_data is not None:
            merged_render = {**render_data, **self.render_data}
        else:
            # 处理其中一个或两者都为 None 的情况
            merged_render = render_data or self.render_data or {}


        self._exec_code = dreams_query_template.render(merged_render)
        logger.info(f'{self.__class__},生成代码成功 {self.calculate_md5()}')
        return self._exec_code


class AIProgramGenerator(CodeGenerator):

    _exec_code: str

    def __init__(self, ai_code_file: str, render_data: dict = {}):
        super().__init__()
        self.render_data = render_data
        # 读取模板文件
        with open(ai_code_file, 'r') as ai_template_file:
            self.ai_template_content = ai_template_file.read()

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        ai_code_file = cfg.get("ai_code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(ai_code_file=get_template_path(ai_code_file), render_data=render_data)

    @property
    def render_code(self) -> str:
        return self._exec_code

    def generate(self, render_data: dict = {}) -> str:
        logger.info(f'{self.__class__},生成代码')
        ai_template = Template(self.ai_template_content)
        if render_data is not None and self.render_data is not None:
            merged_render = {**render_data, **self.render_data}
        else:
            # 处理其中一个或两者都为 None 的情况
            merged_render = render_data or self.render_data or {}


        self._exec_code = ai_template.render(merged_render)
        logger.info(f'{self.__class__},生成代码成功 {self.calculate_md5()}')
        return self._exec_code


class EngineProgramGenerator(CodeGenerator):

    _exec_code: str

    def __init__(self,engine_code_file: str, render_data: dict = {}):
        super().__init__()
        self.render_data = render_data
        # 读取模板文件
        with open(engine_code_file, 'r') as engine_template_file:
            self.engine_template_content = engine_template_file.read()

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            raise RuntimeError("from_config cfg is None.")
        engine_code_file = cfg.get("engine_code_file", "")
        render_data = cfg.get("render_data", None)
        return cls(engine_code_file=get_template_path(engine_code_file), render_data=render_data)

    @property
    def render_code(self) -> str:
        return self._exec_code

    def generate(self, render_data: dict = {}) -> str:
        logger.info(f'{self.__class__},生成代码')
        engine_template = Template(self.engine_template_content)
        if render_data is not None and self.render_data is not None:
            merged_render = {**render_data, **self.render_data}
        else:
            # 处理其中一个或两者都为 None 的情况
            merged_render = render_data or self.render_data or {}

        self._exec_code = engine_template.render(merged_render)
        logger.info(f'{self.__class__},生成代码成功 {self.calculate_md5()}')
        return self._exec_code
