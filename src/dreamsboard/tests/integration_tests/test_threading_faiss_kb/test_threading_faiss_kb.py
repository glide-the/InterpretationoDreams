from pprint import pprint
from unittest.mock import MagicMock

import pytest

from dreamsboard.vector.base import DocumentWithVSId
from dreamsboard.vector.faiss_kb_service import FaissCollectionService


@pytest.fixture
def faiss_service(embed_model_path: str) -> FaissCollectionService:
    """Fixture to create a FAISS collection bound to the configured embeddings."""

    service = FaissCollectionService(
        kb_name="faiss",
        embed_model=embed_model_path,
        vector_name="samples",
        device="cpu",
    )
    yield service
    service.do_clear_vs()


@pytest.mark.parametrize(
    "query, expected_num_results",
    [
        ("vs1", 1),
        ("vs2", 1),  # Assuming no document for vs2
    ],
)
def test_search_with_parametrization(faiss_service, query, expected_num_results):
    """Test search with parametrization for different query inputs."""

    docs = [DocumentWithVSId(id="1", page_content=f"text added by {query}")]

    docs = faiss_service.do_add_doc(docs)
    pprint(docs)
    docs = faiss_service.do_search(query=f"{query}", top_k=3, score_threshold=0.8)
    pprint(docs)

    faiss_service.do_clear_vs()
    assert len(docs) == expected_num_results
