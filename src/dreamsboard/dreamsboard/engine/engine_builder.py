from langchain.schema import BaseMessage

from dreamsboard.engine.data_structs.data_structs import IndexStruct, IndexDict
from dreamsboard.engine.schema import BaseNode
from dreamsboard.engine.generate.code_executor import CodeExecutor
from dreamsboard.engine.generate.code_generate import CodeGenerator
from dreamsboard.engine.generate.run_generate import CodeGeneratorHandler, CodeGeneratorChain
from typing import Any, Dict, List, Generic, TypeVar, Sequence, Type, Optional
from abc import ABC, abstractmethod

IS = TypeVar("IS", bound=IndexStruct)
EngineBuilderType = TypeVar("EngineBuilderType", bound="BaseEngineBuilder")


class BaseEngineBuilder(Generic[IS], ABC):

    index_struct_cls: Type[IS]

    def __init__(
            self,
            nodes: Optional[Sequence[CodeGenerator]] = None,
            index_struct: Optional[IS] = None,
            show_progress: bool = False,
            **kwargs: Any,
    ) -> None:
        """Initialize with parameters."""
        if index_struct is None and nodes is None:
            raise ValueError("One of nodes or index_struct must be provided.")
        if index_struct is not None and nodes is not None:
            raise ValueError("Only one of nodes or index_struct can be provided.")
        # This is to explicitly make sure that the old UX is not used
        if nodes is not None and len(nodes) >= 1 and not isinstance(nodes[0], CodeGenerator):
            raise ValueError("nodes must be a list of CodeGenerator objects.")

        self._show_progress = show_progress

        if index_struct is None:
            assert nodes is not None
            index_struct = self.build_index_from_nodes(nodes)
        self._index_struct = index_struct

    @classmethod
    def from_message_node(cls, messages: Optional[Sequence[BaseMessage]]) -> EngineBuilderType:
        """Build a CodeGeneratorBuilder from a list of messages."""
        nodes = []
        if messages is not None:
            for message in messages:
                nodes.append(CodeGeneratorHandler.from_message(message))
        return cls(nodes=nodes)

    @abstractmethod
    def _build_index_from_nodes(self, nodes: Sequence[CodeGenerator]) -> IS:
        """Build the index from nodes."""

    def build_index_from_nodes(self, nodes: Sequence[CodeGenerator]) -> IS:

        return self._build_index_from_nodes(nodes)

    @abstractmethod
    def _add_generator(self, nodes: Sequence[CodeGenerator], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""

    def add_generators(self, nodes: Sequence[CodeGenerator], **insert_kwargs: Any) -> None:

        self._add_generator(nodes, **insert_kwargs)

    def add_generator(self, node: CodeGenerator, **insert_kwargs: Any) -> None:
        """Insert a document."""

        self.add_generators([node], **insert_kwargs)


# Create a code generator builder
class CodeGeneratorBuilder(BaseEngineBuilder[IndexDict]):
    """
    BaseEngineBuilder[IndexDict]
    BaseEngineBuilder: 构建器基类，用于构建代码生成器的构建器.
    IndexDict: 用于存储索引的数据结构,支持链式调用、序列化和反序列化
    code_gen_chain: 代码链接器，用于存储代码生成器
    用于构建代码生成器的构建器，支持链式调用、序列化和反序列化
    被链接起来的代码生成器会按照添加的顺序依次执行，生成最终的代码
    连接器可被序列化，用于保存和加载
    """
    index_struct_cls = IndexDict
    _code_gen_chain: CodeGeneratorChain = CodeGeneratorChain()

    def _build_index_from_nodes(self, nodes: Sequence[CodeGenerator]) -> IndexDict:
        """Build index from nodes."""
        index_struct = self.index_struct_cls()
        self._add_nodes_to_index(
            index_struct, nodes, show_progress=self._show_progress
        )

        return index_struct

    def _add_generator(self, nodes: Sequence[CodeGenerator], **insert_kwargs: Any) -> None:

        self._add_nodes_to_index(self._index_struct, nodes, **insert_kwargs)

    def _add_nodes_to_index(
            self,
            index_struct: IndexDict,
            nodes: Sequence[CodeGenerator],
            show_progress: bool = False,
    ) -> None:
        """
        添加节点到索引中，节点的
        :param index_struct:
        :param nodes:
        :param show_progress:
        :return:
        """
        if not nodes:
            return

        for node in nodes:

            self._code_gen_chain.add_generator(node)
            # NOTE: remove embedding from node to avoid duplication
            node_without_embedding = node.copy()
            node_without_embedding.embedding = None

            index_struct.add_node(node_without_embedding)

    def remove_last_generator(self):
        self._index_struct.delete(doc_id=self._code_gen_chain.chain_tail.generator.node_id)
        self._code_gen_chain.remove_last_generator()

    @property
    def summary(self) -> str:
        return str(self._index_struct.summary)

    @summary.setter
    def summary(self, new_summary: str) -> None:
        self._index_struct.summary = new_summary

    def build_executor(self, render_data: dict = {}) -> CodeExecutor:
        if self._code_gen_chain.chain_head is None:
            raise RuntimeError("chain_head is None.")

        executor_code = self._code_gen_chain.generate(render_data)
        self.summary = executor_code
        executor = CodeExecutor(executor_code=executor_code)
        return executor
