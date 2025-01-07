import os
import fsspec
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from typing import Any, Dict, List, Optional, Sequence

from dreamsboard.engine.entity.task_step.task_step import TaskStepNode

DEFAULT_PERSIST_FNAME = "task_step_store.json"
DEFAULT_PERSIST_DIR = "./storage"
DEFAULT_PERSIST_PATH = os.path.join(DEFAULT_PERSIST_DIR, DEFAULT_PERSIST_FNAME)


@dataclass
class RefTaskStepInfo(DataClassJsonMixin):
    """Dataclass to represent ingested task steps."""

    node_ids: List = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTaskStepStore(ABC):
    # ===== ，加载与保存抽象定义Save/load =====
    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the docstore to a file."""
        pass

    # ===== 存储任务步骤相关的信息 =====
    @property
    @abstractmethod
    def task_step_all(self) -> Dict[str, TaskStepNode]:
        ...

    @abstractmethod
    def add_task_step(
        self, task_steps: Sequence[TaskStepNode], allow_update: bool = True
    ) -> None:
        ...

    @abstractmethod
    def get_task_step(self, task_step_id: str, raise_error: bool = True) -> Optional[TaskStepNode]:
        ...

    @abstractmethod
    def delete_task_step(self, task_step_id: str, raise_error: bool = True) -> None:
        """Delete a task_step from the store."""
        ...

    @abstractmethod
    def task_step_exists(self, task_step_id: str) -> bool:
        ...

    # ==== 存储任务步骤相关的信息引用节点 =====
    @abstractmethod
    def get_all_ref_task_step_info(self) -> Optional[Dict[str, RefTaskStepInfo]]:
        """Get a mapping of ref_task_step_id -> RefTaskStepInfo for all ingested nodes."""

    @abstractmethod
    def get_ref_task_step_info(self, ref_task_step_id: str) -> Optional[RefTaskStepInfo]:
        """Get the RefTaskStepInfo for a given ref_task_step_id."""

    @abstractmethod
    def delete_ref_task_step(self, ref_task_step_id: str, raise_error: bool = True) -> None:
        """Delete a ref_task_step and all it's associated nodes."""

    @abstractmethod
    def ref_task_step_exists(self, ref_task_step_id: str) -> bool:
        """Check if a ref_task_step_id has been ingested."""
