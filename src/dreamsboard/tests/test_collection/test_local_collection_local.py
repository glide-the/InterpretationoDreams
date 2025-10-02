from typing import List

import pytest

from dreamsboard.collection import create_collection
from dreamsboard.vector.base import DocumentWithVSId


class _DummyFaissService:
    def __init__(self, *args, **kwargs):
        self._docs: List[DocumentWithVSId] = []

    def do_clear_vs(self):
        self._docs.clear()

    def do_add_doc(self, docs):
        self._docs.extend(docs)

    def do_search(self, query: str, top_k: int, score_threshold: float = 1):
        results: List[DocumentWithVSId] = []
        for doc in self._docs:
            metadata = dict(doc.metadata)
            metadata["score"] = metadata.get("score", 1.0)
            results.append(
                DocumentWithVSId(
                    id=doc.id,
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )
        return results[:top_k]

    def get_doc_by_ids(self, ids):
        selected = []
        for doc in self._docs:
            if doc.id in ids:
                selected.append(doc)
        return selected


@pytest.fixture(autouse=True)
def patch_faiss_service(monkeypatch):
    monkeypatch.setattr(
        "dreamsboard.collection.local_collection.FaissCollectionService",
        _DummyFaissService,
    )
    yield


def test_local_collection_from_directory(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.md").write_text("# Heading\n\nSome local content.", encoding="utf-8")

    collection = create_collection(
        "local_collection",
        kb_name="kb",
        embed_model="model",
        vector_name="vectors",
        device="cpu",
        docs_path=str(docs_dir),
        chunk_size=50,
        chunk_overlap=0,
    )
    results = collection.query("local", top_k=2)
    assert results
    assert "local" in results[0].content.lower()
    assert results[0].metadata["ref_id"]
