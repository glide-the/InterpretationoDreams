from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import os

import fsspec

from dreamsboard.engine.constants import (
    DOC_STORE_KEY,
    INDEX_STORE_KEY, DREAMS_ANALYSIS_STORE_KEY,
    TASK_STEP_STORE_KEY
)

from dreamsboard.engine.storage.task_step_store.simple_task_step_store import SimpleTaskStepStore
from dreamsboard.engine.storage.task_step_store.types import BaseTaskStepStore
from dreamsboard.engine.storage.task_step_store.types import DEFAULT_PERSIST_FNAME as TASK_STEP_STORE_FNAME

from dreamsboard.engine.storage.dreams_analysis_store.simple_dreams_analysis_store import SimpleDreamsAnalysisStore
from dreamsboard.engine.storage.dreams_analysis_store.types import BaseDreamsAnalysisStore
from dreamsboard.engine.storage.dreams_analysis_store.types import DEFAULT_PERSIST_FNAME as DREAMS_ANALYSIS_STORE_FNAME
from dreamsboard.engine.storage.template_store.simple_template_store import SimpleTemplateStore
from dreamsboard.engine.storage.template_store.types import DEFAULT_PERSIST_FNAME as TEMPLATE_STORE_FNAME
from dreamsboard.engine.storage.template_store.types import BaseTemplateStore
from dreamsboard.engine.storage.index_store.simple_index_store import SimpleIndexStore
from dreamsboard.engine.storage.index_store.types import (
    DEFAULT_PERSIST_FNAME as INDEX_STORE_FNAME,
)
from dreamsboard.engine.storage.index_store.types import BaseIndexStore
from dreamsboard.engine.utils import concat_dirs

DEFAULT_PERSIST_DIR = "./storage"


@dataclass
class StorageContext:
    """Storage context.

    The storage context container is a utility container for storing nodes,
    indices, and vectors. It contains the following:
    - dreams_analysis_store: BaseTemplateStore
    - template_store: BaseTemplateStore
    - index_store: BaseIndexStore

    """
    task_step_store: BaseTaskStepStore
    dreams_analysis_store: BaseDreamsAnalysisStore
    template_store: BaseTemplateStore
    index_store: BaseIndexStore

    @classmethod
    def from_defaults(
            cls,
            task_step_store: Optional[BaseTaskStepStore] = None,
            dreams_analysis_store: Optional[BaseDreamsAnalysisStore] = None,
            template_store: Optional[BaseTemplateStore] = None,
            index_store: Optional[BaseIndexStore] = None,
            persist_dir: Optional[str] = None,
            fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> "StorageContext":
        """Create a StorageContext from defaults.

        Args:
            task_step_store (Optional[BaseTaskStepStore]): task_step_store
            dreams_analysis_store (Optional[BaseDreamsAnalysisStore]): dreams_analysis_store
            template_store (Optional[BaseTemplateStore]): document store
            index_store (Optional[BaseIndexStore]): index store
            :param dreams_analysis_store:
            :param template_store:
            :param index_store:
            :param fs:
            :param persist_dir:

        """
        if persist_dir is None:
            task_step_store = task_step_store or SimpleTaskStepStore()
            dreams_analysis_store = dreams_analysis_store or SimpleDreamsAnalysisStore()
            template_store = template_store or SimpleTemplateStore()
            index_store = index_store or SimpleIndexStore()
        else:
            task_step_store = task_step_store or SimpleTaskStepStore.from_persist_dir(
                persist_dir, fs=fs
            )
            dreams_analysis_store = dreams_analysis_store or SimpleDreamsAnalysisStore.from_persist_dir(
                persist_dir, fs=fs
            )
            template_store = template_store or SimpleTemplateStore.from_persist_dir(
                persist_dir, fs=fs
            )
            index_store = index_store or SimpleIndexStore.from_persist_dir(
                persist_dir, fs=fs
            )

        return cls(task_step_store, dreams_analysis_store, template_store, index_store)

    def persist(
            self,
            persist_dir: Union[str, os.PathLike] = DEFAULT_PERSIST_DIR,
            task_step_store_fname: str = TASK_STEP_STORE_FNAME,
            dreams_analysis_store_fname: str = DREAMS_ANALYSIS_STORE_FNAME,
            template_store_fname: str = TEMPLATE_STORE_FNAME,
            index_store_fname: str = INDEX_STORE_FNAME,
            fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the storage context.

        Args:
            persist_dir (str): directory to persist the storage context
            task_step_store_fname (str): filename for the task_step_store
            dreams_analysis_store_fname (str): filename for the dreams_analysis_store
            template_store_fname (str): filename for the template_store
            index_store_fname (str): filename for the index_store
            fs (Optional[fsspec.AbstractFileSystem]): filesystem to use

        """
        if fs is not None:
            persist_dir = str(persist_dir)  # NOTE: doesn't support Windows here
            task_step_store_path = concat_dirs(persist_dir, task_step_store_fname)
            dreams_analysis_store_path = concat_dirs(persist_dir, dreams_analysis_store_fname)
            template_store_path = concat_dirs(persist_dir, template_store_fname)
            index_store_path = concat_dirs(persist_dir, index_store_fname)
        else:
            persist_dir = Path(persist_dir)
            task_step_store_path = str(persist_dir / task_step_store_fname)
            dreams_analysis_store_path = str(persist_dir / dreams_analysis_store_fname)
            template_store_path = str(persist_dir / template_store_fname)
            index_store_path = str(persist_dir / index_store_fname)

        self.task_step_store.persist(persist_path=task_step_store_path, fs=fs)
        self.dreams_analysis_store.persist(persist_path=dreams_analysis_store_path, fs=fs)
        self.template_store.persist(persist_path=template_store_path, fs=fs)
        self.index_store.persist(persist_path=index_store_path, fs=fs)

    def to_dict(self) -> dict:
        all_simple = (
                isinstance(self.task_step_store, SimpleTaskStepStore)
                and isinstance(self.dreams_analysis_store, SimpleDreamsAnalysisStore)
                and isinstance(self.template_store, SimpleTemplateStore)
                and isinstance(self.index_store, SimpleIndexStore)
        )
        if not all_simple:
            raise ValueError(
                "to_dict only available when using simple doc/index/vector stores"
            )

        assert isinstance(self.task_step_store, SimpleTaskStepStore)
        assert isinstance(self.dreams_analysis_store, SimpleDreamsAnalysisStore)
        assert isinstance(self.template_store, SimpleTemplateStore)
        assert isinstance(self.index_store, SimpleIndexStore)

        return {
            TASK_STEP_STORE_KEY: self.task_step_store.to_dict(),
            DREAMS_ANALYSIS_STORE_KEY: self.dreams_analysis_store.to_dict(),
            DOC_STORE_KEY: self.template_store.to_dict(),
            INDEX_STORE_KEY: self.index_store.to_dict(),
        }

    @classmethod
    def from_dict(cls, save_dict: dict) -> "StorageContext":
        """Create a StorageContext from dict."""
        task_step_store = SimpleTaskStepStore.from_dict(save_dict[TASK_STEP_STORE_KEY])
        dreams_analysis_store = SimpleDreamsAnalysisStore.from_dict(save_dict[DREAMS_ANALYSIS_STORE_KEY])
        template_store = SimpleTemplateStore.from_dict(save_dict[DOC_STORE_KEY])
        index_store = SimpleIndexStore.from_dict(save_dict[INDEX_STORE_KEY])
        return cls(
            task_step_store=task_step_store,
            dreams_analysis_store=dreams_analysis_store,
            template_store=template_store,
            index_store=index_store,
        )
