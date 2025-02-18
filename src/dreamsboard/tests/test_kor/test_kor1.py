import os

from kor.extraction import create_extraction_chain
from kor.nodes import Number, Object, Text
from langchain_community.chat_models import ChatOpenAI


def test_kor():
    llm = ChatOpenAI(
        openai_api_base="http://127.0.0.1:30000/v1",
        model="glm-3-turbo",
        openai_api_key="glm-4",
        verbose=True,
        temperature=0.95,
        top_p=0.70,
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
                """嗯,这是一个复杂的职业和个人境况。作为心理咨询师,我会以开放和理解的态度来引导来访者谈论他的感受和想法。以下是我会采取的一些步骤:
    
    第一步,我会建立信任关系,让来访者感到被理解和尊重。我会说类似:“我理解这对你来说是一个困难的处境。我在这里倾听,请尽可能开放地和我分享你的想法。”
    
    第二步,我会使用开放式问题来鼓励来访者进一步探讨他的感受。例如:“这份工作升职对你来说意味着什么?”“当你听到这个消息时,你有什么样的情绪反应?”“你觉得自己和R先生处于类似的困境吗?为什么?”
    
    第三步,我会探索来访者的自我评价和自我认知。例如:“你如何看待自己与其他候选人的比较?”“你认为自己配不配得到这个升职机会?”“你对自己的能力和价值有何看法?”
    
    第四步,我会引导来访者思考他的职业目标和动机。例如:“这份工作对你的意义是什么?”“成为教授对你来说意味着什么?”“在当前的环境下,你更倾向于哪种选择?”
    
    第五步,我会鼓励来访者表达对未来和现状的想法。例如:“你理想中的职业道路是什么?”“你觉得现在可以采取哪些行动?”“对你来说,最重要的价值观是什么?”
    
    在整个过程中,我会积极倾听,提供支持和鼓励,并辅以必要的情绪调节技巧,帮助来访者开放地表达自己,获得情感释放。我相信以理解、尊重和同情的态度可以帮助来访者面对当前的职业和情感困境。""",
                [
                    {
                        "step_advice": "我理解这对你来说是一个困难的处境。我在这里倾听,请尽可能开放地和我分享你的想法。",
                        "step_description": "建立信任和舒适感",
                    },
                    {
                        "step_advice": "这份工作升职对你来说意味着什么?",
                        "step_description": "使用开放性问题",
                    },
                    {
                        "step_advice": "当你听到这个消息时,你有什么样的情绪反应?",
                        "step_description": "使用开放性问题",
                    },
                    {
                        "step_advice": "你觉得自己和R先生处于类似的困境吗?",
                        "step_description": "使用开放性问题",
                    },
                    {
                        "step_advice": "你如何看待自己与其他候选人的比较?",
                        "step_description": "探索情感体验",
                    },
                    {
                        "step_advice": "你认为自己配不配得到这个升职机会?",
                        "step_description": "确认自我怀疑",
                    },
                    {
                        "step_advice": "你对自己的能力和价值有何看法?",
                        "step_description": "探索情感体验",
                    },
                    {
                        "step_advice": "这份工作对你的意义是什么?",
                        "step_description": "鼓励深入表达",
                    },
                    {
                        "step_advice": "成为教授对你来说意味着什么?",
                        "step_description": "鼓励深入表达",
                    },
                    {
                        "step_advice": "在当前的环境下,你更倾向于哪种选择?",
                        "step_description": "引导来访者思考他的职业目标和动机",
                    },
                    {
                        "step_advice": "你理想中的职业道路是什么?",
                        "step_description": "创造非评判性环境",
                    },
                    {
                        "step_advice": "你觉得现在可以采取哪些行动?",
                        "step_description": "鼓励自由表达",
                    },
                    {
                        "step_advice": "对你来说,最重要的价值观是什么?",
                        "step_description": "鼓励来访者表达对未来和现状的想法",
                    },
                ],
            )
        ],
        many=True,
    )

    chain = create_extraction_chain(llm, schema)
    print(chain.prompt.format_prompt(text="[user input]").to_string())

    response = chain.run(
        """以下是如何通过分步引导来访者张毛峰说出自己的问题的步骤：

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

通过这样的引导，可以帮助张毛峰表达和认识自己的情感、动机和内心冲突，从而更好地理解自己的问题所在。
    """
    )

    for chat in response["data"]["script"]:
        print(chat)
