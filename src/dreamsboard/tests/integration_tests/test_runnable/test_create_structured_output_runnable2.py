import logging
from typing import List, Optional

import langchain
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from dreamsboard.document_loaders.protocol.ner_protocol import (
    DreamsStepInfo,
    DreamsStepInfoListWrapper,
)

langchain.verbose = True
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
# 控制台打印
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

logger.addHandler(handler)


def test_create_structured_output_runnable2() -> None:
    """Test create_structured_output_runnable. 测试创建结构化输出可运行对象。"""
    llm = ChatOpenAI(
        openai_api_base="http://127.0.0.1:30000/v1",
        model="glm-4",
        openai_api_key="glm-4",
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """根据提供的故事场景，您作为心理咨询工作者可以使用开放性问题来引导来访者表达他们的感受和思维。以下是一步一步的分解：

**Step 1: 建立情感连接**
开始时，您可以通过表达理解和共鸣来建立情感连接，让来访者感到舒适。您可以说：“我注意到这个对话中有许多愉快的时刻和互动。你对这些时刻有什么特别的感受吗？”

**Step 2: 探索来访者的感受**
继续引导来访者表达他们的感受。您可以问：“在这个对话中，有哪些瞬间让你感到开心或快乐？”

**Step 3: 询问是否有反感情绪**
除了积极的情感，也要问询是否有一些负面情感或担忧。您可以说：“除了快乐的瞬间，是否有一些让你感到不安或担忧的地方？”

**Step 4: 深入探讨个人反应**
一旦来访者开始分享他们的感受，可以深入探讨他们的个人反应。例如：“你觉得自己在这些互动中扮演了什么角色？”

**Step 5: 探索与他人的互动**
继续引导来访者思考他们与他人的互动。您可以问：“这些互动对你与他人的关系有什么影响？你觉得与朋友之间的互动如何影响你的情感状态？”

**Step 6: 引导自我反思**
最后，鼓励来访者进行自我反思。您可以问：“在这个故事场景中，你是否注意到了自己的情感变化或思维模式？有没有什么你想要深入探讨或解决的问题？”

通过这种方式，您可以引导来访者自由表达他们的情感和思维，帮助他们更好地理解自己和他们与他人的互动。同时，保持开放和倾听，以便在需要时提供支持和建议。""",
            ),
            (
                "ai",
                """step_advice|step_description
我注意到这个对话中有许多愉快的时刻和互动。你对这些时刻有什么特别的感受吗？|建立情感连接
在这个对话中，有哪些瞬间让你感到开心或快乐?|探索来访者的感受
除了快乐的瞬间，是否有一些让你感到不安或担忧的地方？|询问是否有反感情绪
你觉得自己在这些互动中扮演了什么角色?|深入探讨个人反应
这些互动对你与他人的关系有什么影响？你觉得与朋友之间的互动如何影响你的情感状态?|探索与他人的互动
在这个故事场景中，你是否注意到了自己的情感变化或思维模式？有没有什么你想要深入探讨或解决的问题?|引导自我反思""",
            ),
            ("human", "{input}"),
        ]
    )
    chain = create_structured_output_runnable(DreamsStepInfoListWrapper, llm, prompt)
    out = chain.invoke(
        {
            "input": """以下是如何通过分步引导来访者张毛峰说出自己的问题的步骤：

### Step 1: 建立情感连接
- “张毛峰，我注意到你提到的这段经历让你感到非常愉快，甚至有些事情让你忍不住笑出声。能跟我分享一下，那个让你这么开心的事情吗？”

### Step 2: 探索来访者的积极感受
- “在那次和小姐姐的互动中，有哪些具体的瞬间让你感到特别快乐？”

### Step 3: 引导来访者表达背后的动机和情感
- “我看到你因为举报那位小姐姐而感到开心，这背后有没有什么特别的原因？你能告诉我举报她的行为对你意味着什么吗？”

### Step 4: 探询来访者的内心冲突和不安
- “虽然这件事情让你觉得开心，但我想知道，在整件事情中，有没有什么让你感到困扰或不安的地方？”

### Step 5: 询问来访者的反思和感受
- “你提到觉得有时候没有必要讲道理，这是不是说明这件事情也让你感到某种程度上的矛盾和困惑？”

### Step 6: 深入探讨来访者的自我认知
- “你说‘我不想玩了，因为我不是我自己了’，这句话似乎透露出你对自己有一些新的认识。你能解释一下这是什么意思吗？”

以下是具体的步骤分解：

**Step 1: 建立情感连接**
- 通过询问张毛峰感到愉快的时刻来建立情感上的联系。

**Step 2: 探索来访者的积极感受**
- 帮助张毛峰回忆和描述那些让他感到快乐的细节。

**Step 3: 引导来访者表达背后的动机和情感**
- 深入探讨举报行为背后的动机和情感，了解他为何会因为举报而感到快乐。

**Step 4: 探询来访者的内心冲突和不安**
- 询问是否有任何不安或冲突的情绪，比如对于自己行为的道德考量。

**Step 5: 询问来访者的反思和感受**
- 让张毛峰反思他的行为和感受，了解他对于讲道理的看法。

**Step 6: 深入探讨来访者的自我认知**
- 针对他提到的“我不是我自己了”这句话，探索他的自我认同和可能的内心变化。

通过这样的引导，可以帮助张毛峰表达和认识自己的情感、动机和内心冲突，从而更好地理解自己的问题所在。"""
        }
    )

    logger.info(out)
    # -> Dog(name="Harry", color="brown", fav_food="chicken")
