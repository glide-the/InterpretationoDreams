

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

import os
import threading

from dreamsboard.common.callback import (event_manager)
logger = logging.getLogger(__name__)

 
def test_query_database():
    
    collection_id = get_query_hash("test_loader_into_database")
    embed_model_path = "/mnt/ceph/develop/jiawei/model_checkpoint/m3e-base"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    collection = FaissCollectionService(
        kb_name=collection_id,
        embed_model=embed_model_path,
        vector_name="samples",
        device=device
    ) 
   
    owner = f"register_event thread {threading.get_native_id()}"
    logger.info(f"owner:{owner}")
    # 召回
    response = collection.do_search(
        query="上市公司重大资产重组管理办法第十四条", top_k=10, score_threshold=0.6
    )
    print(response)

