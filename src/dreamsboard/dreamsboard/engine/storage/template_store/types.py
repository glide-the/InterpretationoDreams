import os
import fsspec
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from typing import Any, Dict, List, Optional, Sequence

from dreamsboard.engine.schema import BaseNode


DEFAULT_PERSIST_FNAME = "template_store.json"
DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_PATH = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME)


@dataclass
class RefTemplateInfo(DataClassJsonMixin):
    """Dataclass to represent ingested templates."""

    node_ids: List = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTemplateStore(ABC):
    # ===== Save/load =====
    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the docstore to a file."""
        pass

    # ===== Main interface =====
    @property
    @abstractmethod
    def templates(self) -> Dict[str, BaseNode]:
        ...

    @abstractmethod
    def add_templates(
        self, docs: Sequence[BaseNode], allow_update: bool = True
    ) -> None:
        ...

    @abstractmethod
    def get_template(self, template_id: str, raise_error: bool = True) -> Optional[BaseNode]:
        ...

    @abstractmethod
    def delete_template(self, template_id: str, raise_error: bool = True) -> None:
        """Delete a template from the store."""
        ...

    @abstractmethod
    def template_exists(self, template_id: str) -> bool:
        ...

    # ===== Hash =====
    @abstractmethod
    def set_template_hash(self, template_id: str, template_hash: str) -> None:
        ...

    @abstractmethod
    def get_template_hash(self, template_id: str) -> Optional[str]:
        ...

    # ==== Ref Docs =====
    @abstractmethod
    def get_all_ref_template_info(self) -> Optional[Dict[str, RefTemplateInfo]]:
        """Get a mapping of ref_template_id -> RefTemplateInfo for all ingested templates."""

    @abstractmethod
    def get_ref_template_info(self, ref_template_id: str) -> Optional[RefTemplateInfo]:
        """Get the RefTemplateInfo for a given ref_template_id."""

    @abstractmethod
    def delete_ref_template(self, ref_template_id: str, raise_error: bool = True) -> None:
        """Delete a ref_doc and all it's associated nodes."""

    # ===== Nodes =====
    def get_nodes(
        self, node_ids: List[str], raise_error: bool = True
    ) -> List[BaseNode]:
        """Get nodes from docstore.

        Args:
            node_ids (List[str]): node ids
            raise_error (bool): raise error if node_id not found

        """
        return [self.get_node(node_id, raise_error=raise_error) for node_id in node_ids]

    def get_node(self, node_id: str, raise_error: bool = True) -> BaseNode:
        """Get node from docstore.

        Args:
            node_id (str): node id
            raise_error (bool): raise error if node_id not found

        """
        doc = self.get_template(node_id, raise_error=raise_error)
        if not isinstance(doc, BaseNode):
            raise ValueError(f"template {node_id} is not a Node.")
        return doc

    def get_node_dict(self, node_id_dict: Dict[int, str]) -> Dict[int, BaseNode]:
        """Get node dict from docstore given a mapping of index to node ids.

        Args:
            node_id_dict (Dict[int, str]): mapping of index to node ids

        """
        return {
            index: self.get_node(node_id) for index, node_id in node_id_dict.items()
        }
