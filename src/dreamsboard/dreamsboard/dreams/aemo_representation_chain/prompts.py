"""

"""

# 00-判断情感表征是否符合.txt
AEMO_REPRESENTATION_PROMPT_TEMPLATE = """作为一个社会学研究学者，您已经查阅了《作为激情的爱情》卢曼编写的书籍，尝试通过参考文献中定义的爱情语义学，总结下方片段
社会学研究相关内容分成如下步骤
研究交流媒介领域的语义信息
研究激情的非理性与风雅情术的偶然性
研究自身的快感是否转移到社会行为上
研究语义信息的固定形式与预期落空因果性，是否存在可激发性拓展到否定物之中
你可以尝试分步思考然后告诉我答案，Step by Step Decomposition



{start_task_context}
"""
 