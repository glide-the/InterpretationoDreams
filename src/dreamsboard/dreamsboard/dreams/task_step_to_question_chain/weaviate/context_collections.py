import os

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter
from weaviate.client import WeaviateAsyncClient, WeaviateClient


def init_context_connect() -> WeaviateClient:
    headers = {"X-Zhipuai-Api-Key": os.environ.get("ZHIPUAI_API_KEY")}
    client = weaviate.connect_to_local(
        headers=headers,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=12000, insert=120)  # Values in seconds
        ),
    )

    print(client.is_ready())
    return client


def init_context_collections(
    client: WeaviateClient, collection_id: str
) -> tuple[str, str]:
    from datetime import datetime

    # Collection name with timestamp
    collection_name = "AtomgitPapers" + collection_id

    collection_name_context = f"{collection_name}Context"
    # client.collections.delete(collection_name)
    # client.collections.delete(collection_name_context)

    if not client.collections.exists(collection_name):
        client.collections.create(
            collection_name,
            reranker_config=Configure.Reranker.transformers(),
            generative_config=Configure.Generative.zhipuai(
                # These parameters are optional
                model="glm-4-plus",
                max_tokens=500,
                temperature=0.7,
                top_p=0.7,
            ),
            vectorizer_config=[
                # Set another named vector
                Configure.NamedVectors.text2vec_transformers(
                    name="chunk_text",
                    source_properties=["chunk_text"],
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    ),
                ),
            ],
            properties=[  # Define properties
                Property(name="ref_id", data_type=DataType.TEXT),
                Property(name="paper_id", data_type=DataType.TEXT),
                Property(name="paper_title", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.NUMBER),
                Property(name="chunk_text", data_type=DataType.TEXT),
                Property(name="original_filename", data_type=DataType.TEXT),
            ],
        )

    if not client.collections.exists(collection_name_context):
        client.collections.create(
            collection_name_context,
            vectorizer_config=Configure.Vectorizer.text2vec_contextionary(
                vectorize_collection_name=False
            ),
            properties=[  # Define properties
                Property(name="refId", data_type=DataType.TEXT),
                Property(name="paperId", data_type=DataType.TEXT),
                Property(name="chunkId", data_type=DataType.NUMBER),
                Property(name="chunkText", data_type=DataType.TEXT),
            ],
        )

    return collection_name, collection_name_context


def insert_into_database(
    client: WeaviateClient,
    collection_name: str,
    union_id: str,
    properties_list: List[dict],
):
    """
    插入数据到向量数据库,检查唯一
    """

    collection = client.collections.get(collection_name)
    union_ids = [item.get(union_id) for item in properties_list]
    response = collection.query.fetch_objects(
        filters=Filter.by_property(union_id).contains_any(union_ids),
    )

    exist_ids = [o.properties[union_id] for o in response.objects]

    for properties in properties_list:
        if properties[union_id] not in exist_ids:
            collection.data.insert(properties=properties)
