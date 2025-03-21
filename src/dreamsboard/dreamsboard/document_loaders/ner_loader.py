from __future__ import annotations

from langchain.chains import LLMChain
from langchain.chains.ernie_functions import create_structured_output_runnable
from langchain.schema.language_model import BaseLanguageModel
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from dreamsboard.document_loaders.protocol.ner_protocol import Personality


class NerLoader:
    """实体关系抽取"""

    @classmethod
    def form_ner_dreams_personality_builder(
        cls, llm_runable: Runnable[LanguageModelInput, BaseMessage]
    ) -> LLMChain:
        """
        生成性格分析的抽取链
        :param llm:
        :return:
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """根据您提供的信息，您的性格特点可以总结如下：
        
1. 热情和温柔：您在描述天气和气氛时使用了"温柔长裙风"这样的形容词，表现出您对温暖和舒适的情感。

2. 情感表达：您在文本中表达了对一个叫"宝宝"的角色的期待和关心，这显示了您的感性和情感表达能力。

3. 好奇心和幽默感：您提到了要做大胆的事情，并且以"嘻嘻"结束，这暗示了您对新奇事物的好奇心和幽默感。

4. 关心家人和亲情：您提到了弟弟给了三颗糖，表现出您关心家人的情感。

5. 乐于分享和帮助：您提到要给宝宝剥虾并询问宝宝是否想知道小鱼在说什么，显示出您愿意分享和帮助他人的特点。

6. 可能有一些难以理解的部分：在文本中也出现了一些不太清楚的情节，如呼救情节和提到"小肚小肚"，这可能表现出您的思维有时候会有些混乱或不太连贯。

总的来说，您的性格特点包括热情、情感表达能力、好奇心、幽默感、亲情关怀以及乐于分享和帮助他人。
            
            """,
                ),
                (
                    "ai",
                    """personality
热情、情感表达能力、好奇心、幽默感、亲情关怀以及乐于分享和帮助他人""",
                ),
                ("human", "{input}"),
            ]
        )
        chain = create_structured_output_runnable(Personality, llm_runable, prompt)
        return chain
