from typing import Optional

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import logging
import langchain

langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


class Company(BaseModel):
    """Identifying information about a Company."""

    name: str = Field(..., description="The Company name")
    use_persons: str = Field(..., description="Current number of users")
    main_management: Optional[str] = Field(None, description="Enterprise business、 Main business")


def test_create_structured_output_runnable() -> None:
    """Test create_structured_output_runnable. 测试创建结构化输出可运行对象。"""
    llm = ChatOpenAI(openai_api_base='http://127.0.0.1:30000/v1',
                     model="glm-4",
                     openai_api_key="glm-4",
                     verbose=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a world class algorithm for extracting information in structured formats."),
            ("human", "Use the given format to extract information from the following input: {input}"),
            ("human", "Tip: Make sure to answer in the correct format"),
        ]
    )
    chain = create_structured_output_runnable(Company, llm, prompt)
    out = chain.invoke({"input": "智谱AI是由清华大学计算机系技术成果转化而来的公司，致力于打造新一代认知智能通用模型，目前有千万人使用。"})

    logger.info(out)
    # -> Dog(name="Harry", color="brown", fav_food="chicken")
