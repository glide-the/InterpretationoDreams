from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import pytest


class _DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def rank(self, *args, **kwargs):  # pragma: no cover - not used here
        return []

 
from dreamsboard.collection import BaseCollection
from dreamsboard.dreams.builder_task_step.base import StructuredTaskStepStoryboard


class _DummyCollection(BaseCollection):
    def __init__(self):
        self.queries: List[str] = []

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, str]]] = None,
    ) -> None:  # pragma: no cover - interface requirement
        return None

    def query(self, query: str, top_k: int = 5):
        self.queries.append(query)
        return []


class _DummyTaskStepStore:
    task_step_all: Dict[str, object] = {}

    def add_task_step(self, _):  # pragma: no cover - not used in unit test
        return None

    def persist(self, persist_path: str):  # pragma: no cover - not used
        return None


class _DummyAemoChain:
    def invoke_aemo_representation_context(self):
        return {"aemo_representation_context": "context"}


@pytest.mark.parametrize(
    "data_base, extra_kwargs",
    [
        ("search_papers", {}),
        ("web_search", {}),
        ("local_collection", {"docs_path": "dummy"}),
    ],
)
def test_form_builder_uses_collection(monkeypatch, data_base, extra_kwargs):
 
    created: Dict[str, Dict[str, object]] = {}

    def fake_create(mode: str, **kwargs):
        created[mode] = kwargs
        return _DummyCollection()

    monkeypatch.setattr(
        "dreamsboard.dreams.builder_task_step.base.create_collection", fake_create
    )
    monkeypatch.setattr(
        "dreamsboard.dreams.builder_task_step.base.AEMORepresentationChain.from_aemo_representation_chain",
        classmethod(lambda *args, **kwargs: _DummyAemoChain()),
    )
    monkeypatch.setattr(
        "dreamsboard.dreams.builder_task_step.base.CrossEncoder",
        _DummyCrossEncoder,
    )
    monkeypatch.setattr(
        "dreamsboard.dreams.builder_task_step.base.SimpleTaskStepStore.from_persist_dir",
        classmethod(lambda cls, *_args, **_kwargs: _DummyTaskStepStore()),
    )

    builder = StructuredTaskStepStoryboard.form_builder(
        llm_runable=None,
        kor_dreams_task_step_llm=None,
        start_task_context="context",
        cross_encoder_path="cross",
        embed_model_path="embed",
        data_base=data_base,
        collection_kwargs=extra_kwargs,
    )

    expected_mode = "web_search" if data_base == "searx" else data_base
    assert expected_mode in created
    call_kwargs = created[expected_mode]
    assert call_kwargs["kb_name"]
    if data_base == "local_collection":
        assert call_kwargs["docs_path"] == "dummy"
    assert isinstance(builder.collection, _DummyCollection)
