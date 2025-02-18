import os

from kor.extraction import create_extraction_chain
from kor.nodes import Number, Object, Text
from langchain_community.chat_models import ChatOpenAI


def test_kor2():
    llm = ChatOpenAI(
        openai_api_base="http://127.0.0.1:30000/v1",
        model="glm-4",
        openai_api_key="glm-4",
        verbose=True,
        temperature=0.1,
        top_p=0.9,
    )
    # @title 长的prompt
    schema = Object(
        id="script",
        description="Adapted from the novel into script",
        attributes=[
            Text(
                id="step_advice",
                description="""Advice provided in this step, e.g. "I would say something like: 'I understand this is a difficult situation for you.'" """,
            ),
            Text(
                id="step_description",
                description="""(Description of the counseling step, e.g. "Establish trust" """,
            ),
        ],
        examples=[
            (
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
                [
                    {
                        "step_advice": "我注意到这个对话中有许多愉快的时刻和互动。你对这些时刻有什么特别的感受吗？",
                        "step_description": "建立情感连接",
                    },
                    {
                        "step_advice": "在这个对话中，有哪些瞬间让你感到开心或快乐?",
                        "step_description": "探索来访者的感受",
                    },
                    {
                        "step_advice": "除了快乐的瞬间，是否有一些让你感到不安或担忧的地方？",
                        "step_description": "询问是否有反感情绪",
                    },
                    {
                        "step_advice": "你觉得自己在这些互动中扮演了什么角色?",
                        "step_description": "深入探讨个人反应",
                    },
                    {
                        "step_advice": "这些互动对你与他人的关系有什么影响？你觉得与朋友之间的互动如何影响你的情感状态?",
                        "step_description": "探索与他人的互动",
                    },
                    {
                        "step_advice": "在这个故事场景中，你是否注意到了自己的情感变化或思维模式？有没有什么你想要深入探讨或解决的问题?",
                        "step_description": "引导自我反思",
                    },
                ],
            )
        ],
        many=True,
    )

    chain = create_extraction_chain(llm, schema)
    print(chain.prompt.format_prompt(text="[user input]").to_string())

    response = chain.run(
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

通过这种方式，您可以引导来访者自由表达他们的情感和思维，帮助他们更好地理解自己和他们与他人的互动。同时，保持开放和倾听，以便在需要时提供支持和建议。
    """
    )

    print(response)
