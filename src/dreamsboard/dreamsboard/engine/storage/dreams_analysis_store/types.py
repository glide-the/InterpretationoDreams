import os
import fsspec
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from typing import Any, Dict, List, Optional, Sequence

from dreamsboard.engine.entity.dreams_personality.dreams_personality import DreamsPersonalityNode

DEFAULT_PERSIST_FNAME = "dreams_analysis_store.json"
DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_PATH = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME)


@dataclass
class RefAnalysisInfo(DataClassJsonMixin):
    """Dataclass to represent ingested analyses."""

    node_ids: List = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDreamsAnalysisStore(ABC):
    # ===== 心里分析后开放问题、性格信息，加载与保存抽象定义Save/load =====
    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the docstore to a file."""
        pass

    # ===== 存储性格相关的信息 =====
    @property
    @abstractmethod
    def analysis_all(self) -> Dict[str, DreamsPersonalityNode]:
        ...

    @abstractmethod
    def add_analysis(
        self, analyses: Sequence[DreamsPersonalityNode], allow_update: bool = True
    ) -> None:
        ...

    @abstractmethod
    def get_analysis(self, analysis_id: str, raise_error: bool = True) -> Optional[DreamsPersonalityNode]:
        ...

    @abstractmethod
    def delete_analysis(self, analysis_id: str, raise_error: bool = True) -> None:
        """Delete a analysis from the store."""
        ...

    @abstractmethod
    def analysis_exists(self, analysis_id: str) -> bool:
        ...

    # ==== 存储性格相关的信息引用节点 =====
    @abstractmethod
    def get_all_ref_analysis_info(self) -> Optional[Dict[str, RefAnalysisInfo]]:
        """Get a mapping of ref_analysis_id -> RefAnalysisInfo for all ingested nodes."""

    @abstractmethod
    def get_ref_analysis_info(self, ref_analysis_id: str) -> Optional[RefAnalysisInfo]:
        """Get the RefAnalysisInfo for a given ref_analysis_id."""

    @abstractmethod
    def delete_ref_analysis(self, ref_analysis_id: str, raise_error: bool = True) -> None:
        """Delete a ref_analysis and all it's associated nodes."""

    @abstractmethod
    def ref_analysis_exists(self, ref_analysis_id: str) -> bool:
        """Check if a ref_analysis_id has been ingested."""
