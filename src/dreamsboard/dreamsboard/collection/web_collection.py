from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Sequence

from dreamsboard.dreams.task_step_to_question_chain.searx.searx import searx_query
from dreamsboard.vector.base import DocumentWithVSId
from dreamsboard.vector.faiss_kb_service import FaissCollectionService

from .collection import BaseCollection, QueryResult, register_collection


class WebCollection(BaseCollection):
    def __init__(
        self,
        kb_name: str,
        embed_model: str,
        vector_name: str,
        device: str,
    ) -> None:
        self._service = FaissCollectionService(
            kb_name=kb_name,
            embed_model=embed_model,
            vector_name=vector_name,
            device=device,
        )

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        docs = self._build_documents_from_texts(texts, metadatas)
        if docs:
            self._service.do_add_doc(docs)

    def _build_documents_from_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]],
    ) -> List[DocumentWithVSId]:
        metadatas = metadatas or [{}] * len(texts)
        documents: List[DocumentWithVSId] = []
        for text, metadata in zip(texts, metadatas):
            metadata = dict(metadata)
            ref_id = metadata.get("ref_id") or str(uuid.uuid4())
            chunk_id = metadata.get("chunk_id") or ref_id
            metadata.setdefault("paper_title", metadata.get("paper_title", metadata.get("title", "")))
            metadata.setdefault("ref_id", ref_id)
            metadata.setdefault("chunk_id", chunk_id)
            documents.append(
                DocumentWithVSId(
                    id=str(ref_id),
                    page_content=text,
                    metadata=metadata,
                )
            )
        return documents

    def _upsert(self, docs: List[DocumentWithVSId]) -> None:
        if not docs:
            return
        existing = self._service.get_doc_by_ids([doc.id for doc in docs])
        existing_ids = {doc.id for doc in existing}
        new_docs = [doc for doc in docs if doc.id not in existing_ids]
        if new_docs:
            self._service.do_add_doc(new_docs)

    def query(self, query: str, top_k: int = 5) -> List[QueryResult]:
        properties = searx_query(query, top_k)
        docs: List[DocumentWithVSId] = []
        for item in properties:
            metadata = dict(item)
            text = metadata.pop("chunk_text", "")
            ref_id = str(metadata.get("ref_id") or uuid.uuid4())
            chunk_id = str(metadata.get("chunk_id") or ref_id)
            metadata.setdefault("paper_title", metadata.get("paper_title", metadata.get("title", "")))
            metadata.setdefault("ref_id", ref_id)
            metadata.setdefault("chunk_id", chunk_id)
            docs.append(
                DocumentWithVSId(
                    id=ref_id,
                    page_content=text,
                    metadata=metadata,
                )
            )
        self._upsert(docs)
        documents = self._service.do_search(query=query, top_k=top_k)
        results: List[QueryResult] = []
        for doc in documents:
            metadata = dict(doc.metadata)
            score = metadata.get("score", 0.0)
            metadata.setdefault("ref_id", metadata.get("ref_id", doc.id))
            metadata.setdefault("chunk_id", metadata.get("chunk_id", doc.id))
            metadata.setdefault("paper_title", metadata.get("paper_title", metadata.get("title", "")))
            results.append(
                QueryResult(
                    content=doc.page_content,
                    score=score,
                    metadata=metadata,
                )
            )
        return results


register_collection("web_search", WebCollection)
