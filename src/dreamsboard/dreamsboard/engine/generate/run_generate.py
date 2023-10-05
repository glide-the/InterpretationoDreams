from dreamsboard.engine.generate.code_executor import CodeExecutor
from dreamsboard.engine.generate.code_generate import CodeGenerator


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


# Create a code generator builder
class CodeGeneratorChain:
    def __init__(self):
        self.chain_head: CodeGeneratorHandler = None
        self.chain_tail: CodeGeneratorHandler = None  # Track the tail of the chain

    def add_generator(self, generator: CodeGenerator):
        handler = CodeGeneratorHandler(generator)
        if self.chain_head is None:
            self.chain_head = handler
            self.chain_tail = handler
        else:
            self.chain_tail.next_handler = handler  # Update the tail
            self.chain_tail = handler  # Update the new tail

    def remove_last_generator(self):
        if self.chain_head is None:
            return

        if self.chain_head == self.chain_tail:
            # Only one handler in the chain
            self.chain_head = None
            self.chain_tail = None
        else:
            current_handler = self.chain_head
            while current_handler.next_handler != self.chain_tail:
                current_handler = current_handler.next_handler
            current_handler.next_handler = None
            self.chain_tail = current_handler

    def generate(self, render_data: dict):
        if self.chain_head is None:
            raise RuntimeError("chain_head is None.")

        return self.chain_head.generate(render_data)
