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
            description="Adapted from the novel into script",
            attributes=[
                Text(
                    id="step_advice",
                    description='''Advice provided in this step, e.g. "I would say something like: 'I understand this is a difficult situation for you.'" ''',
                ),
                Text(
                    id="step_description",
                    description="""(Description of the counseling step, e.g. "Establish trust" """,
                )
            ],
            examples=[
                (
                    """嗯,这是一个复杂的职业和个人境况。作为心理咨询师,我会以开放和理解的态度来引导患者谈论他的感受和想法。以下是我会采取的一些步骤:
            
            第一步,我会建立信任关系,让患者感到被理解和尊重。我会说类似:“我理解这对你来说是一个困难的处境。我在这里倾听,请尽可能开放地和我分享你的想法。”
            
            第二步,我会使用开放式问题来鼓励患者进一步探讨他的感受。例如:“这份工作升职对你来说意味着什么?”“当你听到这个消息时,你有什么样的情绪反应?”“你觉得自己和R先生处于类似的困境吗?为什么?”
            
            第三步,我会探索患者的自我评价和自我认知。例如:“你如何看待自己与其他候选人的比较?”“你认为自己配不配得到这个升职机会?”“你对自己的能力和价值有何看法?”
            
            第四步,我会引导患者思考他的职业目标和动机。例如:“这份工作对你的意义是什么?”“成为教授对你来说意味着什么?”“在当前的环境下,你更倾向于哪种选择?”
            
            第五步,我会鼓励患者表达对未来和现状的想法。例如:“你理想中的职业道路是什么?”“你觉得现在可以采取哪些行动?”“对你来说,最重要的价值观是什么?”
            
            在整个过程中,我会积极倾听,提供支持和鼓励,并辅以必要的情绪调节技巧,帮助患者开放地表达自己,获得情感释放。我相信以理解、尊重和同情的态度可以帮助患者面对当前的职业和情感困境。""",
                    [
                        {"step_advice": "我理解这对你来说是一个困难的处境。我在这里倾听,请尽可能开放地和我分享你的想法。",
                         "step_description": "建立信任和舒适感"},
                        {"step_advice": "这份工作升职对你来说意味着什么?", "step_description": "使用开放性问题"},
                        {"step_advice": "当你听到这个消息时,你有什么样的情绪反应?",
                         "step_description": "使用开放性问题"},
                        {"step_advice": "你觉得自己和R先生处于类似的困境吗?", "step_description": "使用开放性问题"},
                        {"step_advice": "你如何看待自己与其他候选人的比较?", "step_description": "探索情感体验"},
                        {"step_advice": "你认为自己配不配得到这个升职机会?", "step_description": "确认自我怀疑"},
                        {"step_advice": "你对自己的能力和价值有何看法?", "step_description": "探索情感体验"},
                        {"step_advice": "这份工作对你的意义是什么?", "step_description": "鼓励深入表达"},
                        {"step_advice": "成为教授对你来说意味着什么?", "step_description": "鼓励深入表达"},
                        {"step_advice": "在当前的环境下,你更倾向于哪种选择?",
                         "step_description": "引导患者思考他的职业目标和动机"},
                        {"step_advice": "你理想中的职业道路是什么?", "step_description": "创造非评判性环境"},
                        {"step_advice": "你觉得现在可以采取哪些行动?", "step_description": "鼓励自由表达"},
                        {"step_advice": "对你来说,最重要的价值观是什么?",
                         "step_description": "鼓励患者表达对未来和现状的想法"},
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
            description="Adapted from the novel into script",
            attributes=[
                Text(
                    id="personality",
                    description='''Summary of personality traits, e.g. "curiosity, sense of humor" ''',
                )
            ],
            examples=[
                (
                    """根据您提供的信息，您的性格特点可以总结如下：
        
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
                )
            ],
            many=True,
        )
        chain = create_extraction_chain(llm, schema)
        return chain
