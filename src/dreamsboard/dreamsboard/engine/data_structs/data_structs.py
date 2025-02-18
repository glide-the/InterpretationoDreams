import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from dataclasses_json import DataClassJsonMixin

from dreamsboard.engine.data_structs.struct_type import IndexStructType
from dreamsboard.engine.schema import BaseNode


@dataclass
class IndexStruct(DataClassJsonMixin):
    """A base data struct for a LlamaIndex."""

    index_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    summary: Optional[str] = None

    def get_summary(self) -> str:
        """Get text summary."""
        if self.summary is None:
            raise ValueError("summary field of the index_struct not set.")
        return self.summary

    @classmethod
    @abstractmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""


@dataclass
class IndexDict(IndexStruct):
    """A simple dictionary of documents."""

    # TODO: slightly deprecated, should likely be a list or set now
    # mapping from vector store id to node doc_id
    nodes_dict: Dict[str, str] = field(default_factory=dict)

    def add_node(
        self,
        node: BaseNode,
        text_id: Optional[str] = None,
    ) -> str:
        """Add text to table, return current position in list."""
        # # don't worry about child indices for now, nodes are all in order
        # self.nodes_dict[int_id] = node
        vector_id = text_id if text_id is not None else node.node_id
        self.nodes_dict[vector_id] = node.node_id

        return vector_id

    def delete(self, doc_id: str) -> None:
        """Delete a Node."""
        del self.nodes_dict[doc_id]

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.CODE_GENERATOR
