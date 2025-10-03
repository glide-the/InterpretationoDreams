
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


@pytest.mark.parametrize(
    "ref_id, chunk_id, title, link, engines, category, chunk_text",
    [
        ("vs1", 1, "title1", "link1", "engine1", "category1", "chunk_text1"),
        ("vs2", 1, "title2", "link2", "engine2", "category2", "chunk_text2"),
        ("vs3", 1, "title3", "link3", "engine3", "category3", "chunk_text3"),
        ("vs4", 1, "title4", "link4", "engine4", "category4", "chunk_text4"),
        ("vs5", 1, "title5", "link5", "engine5", "category5", "chunk_text5"),
    ],
)
def test_faiss_collection_insert(setup_log, ref_id, chunk_id, title, link, engines, category, chunk_text):

    collection_id = get_query_hash("start_task_context")
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
            "ref_id": ref_id,
            "chunk_id": chunk_id,
            "title": title,
            "link": link,
            "engines": engines,
            "category": category,
            "chunk_text": chunk_text,
        }
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
