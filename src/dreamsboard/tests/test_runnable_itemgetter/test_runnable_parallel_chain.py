from typing import Optional

from langchain.chains.openai_functions import create_structured_output_runnable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import logging
import langchain
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def test_runnable_parallel_chain() -> None:
    """Test create_structured_output_runnable. 测试创建结构化输出可运行对象。"""

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        openai_api_base='http://127.0.0.1:30000/v1',
        model="glm-4",
        openai_api_key="glm-4",
        verbose=True,
        # temperature=0.95,
    )
    joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | llm | StrOutputParser()
    poem_chain = (
            ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | llm
    )

    map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

    out = map_chain.invoke({"topic": "bear"})
    print(out)
