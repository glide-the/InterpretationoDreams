from __future__ import annotations
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel


class KorLoader:

    @classmethod
    def form_kor_dreams_guidance_builder(cls,
                                         llm: BaseLanguageModel) -> LLMChain:
        """
        生成开放问题的抽取链
        :param llm:
        :return:
        """
        # @title 长的prompt
        schema = Object(
            id="script",
            description="开放性引导问题",
            attributes=[
                Text(
                    id="step_advice",
                    description='''在这一步骤中提供的建议，例如“我想说这样的话：‘我理解这对你来说是一个困难的情况。’” ''',
                ),
                Text(
                    id="step_description",
                    description="""咨询步骤的描述，例如“建立信任”""",
                )
            ],
            examples=[
                (
                    """Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.
不要包含"```plaintext"或"```"。


根据提供的故事场景，您作为心理咨询工作者可以使用开放性问题来引导来访者表达他们的感受和思维。以下是一步一步的分解：

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
                        {"step_advice": "我注意到这个对话中有许多愉快的时刻和互动。你对这些时刻有什么特别的感受吗？",
                         "step_description": "建立情感连接"},
                        {"step_advice": "在这个对话中，有哪些瞬间让你感到开心或快乐?", "step_description": "探索来访者的感受"},
                        {"step_advice": "除了快乐的瞬间，是否有一些让你感到不安或担忧的地方？",
                         "step_description": "询问是否有反感情绪"},
                        {"step_advice": "你觉得自己在这些互动中扮演了什么角色?", "step_description": "深入探讨个人反应"},
                        {"step_advice": "这些互动对你与他人的关系有什么影响？你觉得与朋友之间的互动如何影响你的情感状态?", "step_description": "探索与他人的互动"},
                        {"step_advice": "在这个故事场景中，你是否注意到了自己的情感变化或思维模式？有没有什么你想要深入探讨或解决的问题?", "step_description": "引导自我反思"},
                    ],
                )
            ],
            many=True,
        )

        chain = create_extraction_chain(llm, schema)
        return chain

    @classmethod
    def form_kor_dreams_personality_builder(cls,
                                            llm: BaseLanguageModel) -> LLMChain:
        """
        生成性格分析的抽取链
        :param llm:
        :return:
        """
        schema = Object(
            id="script",
            description="性格信息",
            attributes=[
                Text(
                    id="personality",
                    description='''文本中包含的性格评价，例如：“率真和直接、愿意与人互动并享受社交乐趣、对预期之外的情况敏感” ''',
                ),
                Text(
                    id="subj",
                    description='''文本中包含的特定人物，例如：“张毛峰” ''',
                )
            ],
            examples=[
                (
                    """Please output the extracted information in CSV format in Excel dialect. Please use a | as the delimiter. 
 Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.
 不要包含"```plaintext"或"```"。
 
 
 根据您提供的信息，您的性格特点可以总结如下：
        
1. 热情和温柔：您在描述天气和气氛时使用了"温柔长裙风"这样的形容词，表现出您对温暖和舒适的情感。

2. 情感表达：您在文本中表达了对一个叫"宝宝"的角色的期待和关心，这显示了您的感性和情感表达能力。

3. 好奇心和幽默感：您提到了要做大胆的事情，并且以"嘻嘻"结束，这暗示了您对新奇事物的好奇心和幽默感。

4. 关心家人和亲情：您提到了弟弟给了三颗糖，表现出您关心家人的情感。

5. 乐于分享和帮助：您提到要给宝宝剥虾并询问宝宝是否想知道小鱼在说什么，显示出您愿意分享和帮助他人的特点。

6. 可能有一些难以理解的部分：在文本中也出现了一些不太清楚的情节，如呼救情节和提到"小肚小肚"，这可能表现出您的思维有时候会有些混乱或不太连贯。

总的来说，您的性格特点包括热情、情感表达能力、好奇心、幽默感、亲情关怀以及乐于分享和帮助他人。
            
            """,
                    [
                        {"personality": "热情、情感表达能力、好奇心、幽默感、亲情关怀以及乐于分享和帮助他人"}
                    ],
                ),
                (
                    """根据以上分析，以下是对片段中人物性格的总结：

### 张毛峰

- **情感表达直接**：他对视觉刺激有积极的情感反应，并直接表达出来，表现出一种率真和直接的个性。
- **社交积极性**：他的情绪快感转移到社交行为上，提出共同活动的邀请，显示出他是一个愿意与人互动并享受社交乐趣的人。
- **对预期之外的情况敏感**：对话中他对某些意外的反应显示他可能对预期落空的情况有所关注。

### 露ᥫᩣ

- **自我展示的谨慎性**：她发送表情图片后撤回，表明她在自我展示方面可能更加谨慎，对自我形象有着不确定感。
- **情感表达的积极性**：她用词如“好激动”显示她能够在交流中表达积极的情感。
- **对反馈的敏感性**：撤回行为和对“别扭”的提及可能意味着她对社会反馈，尤其是潜在的负面反馈较为敏感。

这些性格特点是从对话中所使用的交流媒介和语义信息中推断出来的，可以为理解他们在社会互动中的行为和反应提供一些洞见。然而，这样的分析仅基于有限的信息，真实性格可能更为复杂。
            
            """,
                    [
                        {"personality": "率真和直接、愿意与人互动并享受社交乐趣、对预期之外的情况敏感", "subj": "张毛峰"},
                        {"personality": "对自我形象有着不确定、能够在交流中表达积极的情感、对社会反馈，尤其是潜在的负面反馈较为敏感", "subj": "露ᥫᩣ"},
                    ],
                ),
                (
                    """根据以上分解，以下是关于片段中人物性格的总结：
### 张毛峰和露ᥫᩣ的性格特征：
1. **社交技巧**：两人都展现出良好的社交技巧，能够通过幽默和轻松的方式进行交流，这表明他们在社交场合中可能是受欢迎和具有亲和力的人。
2. **适应性和灵活性**：他们能够适应数字通信媒介，使用表情符号和图片来丰富交流，显示他们能够适应现代社交方式，并具有一定的灵活性。
3. **共享快感**：通过转账和分享食物照片，两人表现出愿意在社交行为中共享快感，这暗示他们可能是慷慨和注重人际关系的人。
4. **处理误解和不确定性**：在遇到语义信息不清或预期落空时，他们能够以平和的方式处理，这显示了他们的耐心和解决问题的能力。
5. **真实和直接**：当需要澄清事实或情感状态时，他们会直接表达否定物，这表明他们倾向于诚实和直接的沟通方式。
综上所述，张毛峰和露ᥫᩣ在对话中表现出的是一种现代社交网络下成熟的交流方式，他们性格中包含了幽默感、社交技巧、适应性和直接性等特征。这些特点使他们能够在数字媒介主导的社交环境中游刃有余。
            """,
                    [
                        {"personality": "幽默感、社交技巧、适应性和直接性等特征、在社交场合中可能是受欢迎和具有亲和力的人",
                         "subj": "张毛峰"},
                        {"personality":  "幽默感、社交技巧、适应性和直接性等特征、在社交场合中可能是受欢迎和具有亲和力的人、",
                         "subj": "露ᥫᩣ"},
                    ],
                )
            ],
            many=True,
        )
        chain = create_extraction_chain(llm, schema)
        return chain
