

from dreamsboard.engine.storage.storage_context import StorageContext
from langchain_community.chat_models import ChatOpenAI
from dreamsboard.engine.task_engine_builder.core import CodeGeneratorBuilder
from dreamsboard.document_loaders.structured_storyboard_loader import LinkedListNode

from dreamsboard.engine.generate.code_generate import QueryProgramGenerator, BaseProgramGenerator
from langchain.schema import AIMessage 
from langchain.prompts import PromptTemplate

from dreamsboard.engine.memory.mctsr.prompt import (
    gpt_prompt_config,
    RefineResponse,
)
from dreamsboard.common.callback import (event_manager)

import os 
import logging 
import threading
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)

from dreamsboard.vector.faiss_kb_service import FaissCollectionService

from dreamsboard.vector.base import DocumentWithVSId
 

# Example of how to use the EventManager and Iteratorize
def task_function(callback, resource_id, **kwargs):

    print(f"Executing task on resource {resource_id} with args: {kwargs}")
    docs = [DocumentWithVSId(id='1', page_content=f"text added by {resource_id}")]

    response = kwargs.get("faiss_service").get_doc_by_ids(ids=['1'])
    if len(response) == 0 :
        docs = kwargs.get("faiss_service").do_add_doc(docs)
        print(docs)
        docs = kwargs.get("faiss_service").do_search(query=f"{resource_id}", top_k=3, score_threshold=0.8)
        print(docs)
    callback(f"Task result resource_id on resource {resource_id}")
   
    
def test_event_manager(): 
        
    faiss_service = FaissCollectionService(
            kb_name="faiss",
            embed_model="/mnt/ceph/develop/jiawei/model_checkpoint/m3e-base",
            vector_name="samples",
            device="cuda"
    )
    def register_and_execute_events(event_manager, start_idx, end_idx):

        owner = f"start_event thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}")
        event_ids = []
        for i in range(start_idx, end_idx):
                
            owner = f"register_event thread {threading.get_native_id()}"
            logger.info(f"owner:{owner}")
            
            event_id = event_manager.register_event(
                task_function,
                resource_id=f"resource_{i % 2 + 1}",
                kwargs={ 
                    "faiss_service": faiss_service
                },
            )

            event_ids.append(event_id)  # 将 event_id 存储到事件列表中
            
            owner = f"register_event end thread {threading.get_native_id()}"
            logger.info(f"owner:{owner}")
 
        owner = f"event_ids thread {threading.get_native_id()}"
        logger.info(f"owner:{owner}")
        # Retrieve and print the result
        for event_id in event_ids:
                
            results = None
            while results is None or len(results) == 0:
                results = event_manager.get_results(event_id) 
            print(f"Final Results for Event {event_id}: {results}")

    # Thread 1 - Registers and executes events from 0 to 4
    thread1 = threading.Thread(target=register_and_execute_events, args=(event_manager, 0, 5))

    # Thread 2 - Registers and executes events from 5 to 9
    thread2 = threading.Thread(target=register_and_execute_events, args=(event_manager, 5, 10))

    # Start both threads
    thread1.start()
    thread2.start()

    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    print("All events processed successfully.")
 