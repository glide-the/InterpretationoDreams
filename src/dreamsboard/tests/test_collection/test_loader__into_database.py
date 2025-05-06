

from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter

from dreamsboard.dreams.task_step_to_question_chain.weaviate.prepare_load import (
    get_query_hash,
)
from dreamsboard.vector.faiss_kb_service import FaissCollectionService

from dreamsboard.vector.base import CollectionService, DocumentWithVSId
import torch
import logging

import pytest
import threading

from dreamsboard.common.callback import (event_manager)
logger = logging.getLogger(__name__)


def _into_database_query(callback, resource_id, **kwargs) -> None:
    """
    插入数据到向量数据库,检查唯一
    :param union_id_key:  唯一标识
    :param page_content_key:  数据列表
    :param properties_list:  数据列表
    :return: None
    """

    union_ids = [str(item.get(kwargs.get("union_id_key"))) for item in kwargs.get("properties_list")]

    response = kwargs.get("collection").get_doc_by_ids(ids=union_ids)

    exist_ids = [o.metadata[kwargs.get("union_id_key")] for o in response]

    docs = []
    for item in kwargs.get("properties_list"):
        metadata = {key: value for key, value in item.items() if key != kwargs.get("page_content_key")}
        if item.get(kwargs.get("union_id_key")) not in exist_ids:
            doc = DocumentWithVSId(id=item.get(kwargs.get("union_id_key")), page_content=item.get(kwargs.get("page_content_key")), metadata=metadata )
            docs.append(doc)

    doc_infos = kwargs.get("collection").do_add_doc(docs)

    kwargs.get("collection").save_vector_store()
    callback(doc_infos)


def test_loader_into_database():
    loader = DirectoryLoader('C:/Users/Administrator/Desktop/监管',
                             glob="**/*.md",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"},
                             use_multithreading=True,
                             show_progress=True)
    files = loader.load()
    headers_to_split_on = [
        ("#", "head1"),
        ("##", "head2"),
        ("###", "head3"),
        ("####", "head4"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    all_chunks = []
    for docs in files:
        for doc in docs:

            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                if doc.metadata:
                    chunk.metadata["source"] = doc.metadata["source"]

            all_chunks.extend(chunks)

    collection_id = get_query_hash("test_loader_into_database")
    embed_model_path = "D:/model/m3e-base"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collection = FaissCollectionService(
        kb_name=collection_id,
        embed_model=embed_model_path,
        vector_name="samples",
        device=device
    )
    properties_list = [
        {
            "ref_id": get_query_hash(chunk.metadata["head2"]),
            "title": chunk.metadata["head2"],
            "chunk_text": chunk.page_content,
            **chunk.metadata,
        }
        for chunk in all_chunks
    ]
    # 插入数据到数据库
    owner = f"register_event thread {threading.get_native_id()}"
    logger.info(f"owner:{owner}")
    event_id = event_manager.register_event(
        _into_database_query,
        resource_id=f"resource_collection_{collection.kb_name}",
        kwargs={
            "collection": collection,
            "union_id_key": 'ref_id',
            "page_content_key": 'chunk_text',
            "properties_list": properties_list,
        },
    )
    results = None
    while results is None or len(results) == 0:
        results = event_manager.get_results(event_id)
    response = results[0]

    print(response)
