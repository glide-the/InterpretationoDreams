import hashlib
import json
import logging
import os
import tempfile
from typing import List

import requests
from langchain_community.utilities import SearxSearchWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

logger.addHandler(handler)

CACHE_DIR = os.path.join(tempfile.gettempdir(), "query_cache")  # 自定义缓存目录


def ensure_cache_dir():
    """确保缓存目录存在"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def get_query_hash(query):
    """根据 query 生成唯一的哈希值"""
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def get_cache_file_path(query):
    """获取 query 对应的缓存文件路径"""
    query_hash = get_query_hash(query)
    return os.path.join(CACHE_DIR, f"{query_hash}.json")


def check_cache(query):
    """检查是否存在 query 的缓存文件"""
    file_path = get_cache_file_path(query)
    if os.path.exists(file_path):
        logger.info(f"Cache hit file_path: {file_path} for query: {query}")
        with open(file_path, "r", encoding="utf-8") as cache_file:
            return json.load(cache_file)  # 返回缓存内容
    return None


def save_to_cache(query, data):
    """将数据保存到 query 的缓存文件中"""
    ensure_cache_dir()
    file_path = get_cache_file_path(query)
    with open(file_path, "w", encoding="utf-8") as cache_file:
        json.dump(data, cache_file, ensure_ascii=False, indent=4)
        logger.info(f"Data cached for query: {query} at {file_path}")


def searx_query(query, top_k):
    cached_data = check_cache(query)
    if cached_data:
        logger.info(f"Using cached data for query: {query}")
        return cached_data  # 如果有缓存则直接返回
    # 第一步：模糊查询论文
    logger.info(f"Searching papers for query: {query}")

    s = SearxSearchWrapper(engines=['google', 'bing'],searx_host="http://127.0.0.1:8080")

    # {
    #     "snippet": result.get("content", ""),
    #     "title": result["title"],
    #     "link": result["url"],
    #     "engines": result["engines"],
    #     "category": result["category"],
    # }
    results = s.results(
        query=query,
        num_results=top_k,
    )
    logger.info(results)
    api_key = "sk-nnvsTgCojtYSPkD42cn3lA0UZMIrRWMe8dpxP40YRkpCj9wm"
    properties_list = []
    for item in results:
        try:
            details_url = f"https://api.unifuncs.com/api/web-reader/{item.get('link')}?apiKey={api_key}&format=text"
            response = requests.get(details_url)
            json_data = {
                "chunk_text": response.text,
                "ref_id": get_query_hash(item.get("link", "")),
                "chunk_id": get_query_hash(item.get("link", "")),
                "title": item.get("title", ""),
                "link": item.get("link"),
                "engines": item.get("engines"),
                "category": item.get("category"),
            }
            properties_list.append(json_data)
        except Exception as e:
            logger.error(f"Error getting details for {item.get('link')}: {e}")
            continue

    # 保存 properties_list 到临时文件
    save_to_cache(query, properties_list)

    return properties_list
