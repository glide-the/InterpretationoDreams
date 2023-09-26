from dreamsboard.generate.code_executor import CodeExecutor
from dreamsboard.generate.code_generate import CodeGenerator


# 创建一个责任链节点
class CodeGeneratorHandler:
    def __init__(self, generator: CodeGenerator, next_handler=None):
        self.generator = generator
        self.next_handler = next_handler

    def generate(self, render_data: dict):
        code = self.generator.generate(render_data)
        if self.next_handler:
            code += self.next_handler.generate(render_data)
        return code


# 创建一个代码生成器的建造者
class CodeGeneratorBuilder:
    def __init__(self):
        self.chain_head = None

    def add_generator(self, generator: CodeGenerator):
        handler = CodeGeneratorHandler(generator)
        if self.chain_head:
            current_handler = self.chain_head
            while current_handler.next_handler:
                current_handler = current_handler.next_handler
            current_handler.next_handler = handler
        else:
            self.chain_head = handler

    def build_executor(self, render_data: dict):
        if self.chain_head is None:
            raise RuntimeError("chain_head is None.")

        code = self.chain_head.generate(render_data)
        executor = CodeExecutor(code=code)
        return executor
