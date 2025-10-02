from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Sequence
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dreamsboard.vector.base import DocumentWithVSId
from dreamsboard.vector.faiss_kb_service import FaissCollectionService

from .collection import BaseCollection, QueryResult, register_collection


class LocalCollection(BaseCollection):
    """Collection backed by local markdown or text documents."""

    def __init__(
        self,
        kb_name: str,
        embed_model: str,
        vector_name: str,
        device: str,
        docs_path: str,
        glob: str = "**/*.md",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        encoding: str = "utf-8",
    ) -> None:
        self._service = FaissCollectionService(
            kb_name=kb_name,
            embed_model=embed_model,
            vector_name=vector_name,
            device=device,
        )
        self._local_service = FaissCollectionService(
            kb_name=f"{kb_name}_local",
            embed_model=embed_model,
            vector_name=f"{vector_name}_local",
            device=device,
        )
        self._docs_path = docs_path
        self._glob = glob
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._encoding = encoding
        self._indexed = False

    def _ensure_indexed(self) -> None:
        if self._indexed:
            return
        self._index_local_documents()
        self._indexed = True

    def _index_local_documents(self) -> None:
        if not os.path.isdir(self._docs_path):
            raise ValueError(
                f"docs_path '{self._docs_path}' does not exist or is not a directory"
            )

        loader = DirectoryLoader(
            self._docs_path,
            glob=self._glob,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": self._encoding},
            use_multithreading=True,
            show_progress=False,
        )
        raw_documents = loader.load()
        documents = []
        for item in raw_documents:
            if isinstance(item, list):
                documents.extend(item)
            else:
                documents.append(item)
        if not documents:
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        docs_with_ids: List[DocumentWithVSId] = []
        for index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            source = metadata.get("source") or f"local_{index}"
            ref_id = metadata.get("ref_id") or source
            chunk_id = metadata.get("chunk_id") or f"{ref_id}_{index}"
            metadata.setdefault("ref_id", ref_id)
            metadata.setdefault("chunk_id", chunk_id)
            metadata.setdefault("paper_title", os.path.basename(source))
            docs_with_ids.append(
                DocumentWithVSId(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ref_id}_{chunk_id}")),
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )
        if docs_with_ids:
            self._local_service.do_clear_vs()
            self._local_service.do_add_doc(docs_with_ids)
            self._upsert(docs_with_ids)

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self._ensure_indexed()
        docs: List[DocumentWithVSId] = []
        metadatas = metadatas or [{}] * len(texts)
        for text, metadata in zip(texts, metadatas):
            metadata = dict(metadata)
            ref_id = metadata.get("ref_id") or str(uuid.uuid4())
            chunk_id = metadata.get("chunk_id") or ref_id
            metadata.setdefault("ref_id", ref_id)
            metadata.setdefault("chunk_id", chunk_id)
            metadata.setdefault("paper_title", metadata.get("paper_title", ""))
            docs.append(
                DocumentWithVSId(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ref_id}_{chunk_id}")),
                    page_content=text,
                    metadata=metadata,
                )
            )
        if docs:
            self._local_service.do_add_doc(docs)
            self._upsert(docs)

    def _upsert(self, docs: List[DocumentWithVSId]) -> None:
        if not docs:
            return
        existing = self._service.get_doc_by_ids([doc.id for doc in docs])
        existing_ids = {doc.id for doc in existing}
        new_docs = [doc for doc in docs if doc.id not in existing_ids]
        if new_docs:
            self._service.do_add_doc(new_docs)

    def query(self, query: str, top_k: int = 5) -> List[QueryResult]:
        self._ensure_indexed()
        documents = self._local_service.do_search(query=query, top_k=top_k)
        self._upsert(documents)
        documents = self._service.do_search(query=query, top_k=top_k)
        results: List[QueryResult] = []
        for doc in documents:
            metadata = dict(doc.metadata)
            score = metadata.get("score", 0.0)
            metadata.setdefault("ref_id", metadata.get("ref_id", doc.id))
            metadata.setdefault("chunk_id", metadata.get("chunk_id", doc.id))
            metadata.setdefault("paper_title", metadata.get("paper_title", ""))
            results.append(
                QueryResult(
                    content=doc.page_content,
                    score=score,
                    metadata=metadata,
                )
            )
        return results


register_collection("local_collection", LocalCollection)
