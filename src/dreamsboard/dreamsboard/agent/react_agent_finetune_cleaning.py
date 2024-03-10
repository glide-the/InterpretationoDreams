from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import DatasetGenerator
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate, Document


def adapter_agent(llm_agent: OpenAI, query_engine_tools):
    base_agent = ReActAgent.from_tools(query_engine_tools, llm=llm_agent, verbose=True)
    return base_agent


def build_dataset_generator_questions(docs: list[Document], llm: LLM, num=60) -> list[str]:
    base_question_gen_query = (
        "你是一名公司人事/HR，你的任务是安排一次简历筛选。"
        "使用简历提交的文件中提供的上下文， 制定一些问题，"
        "从上下文中捕捉到重要事实形成问题，"
        "将问题限制在所提供的上下文信息内."
        "提取的事实需要验证,不需要标注出具体的来源，例如："
        "有没有人做了知识库的项目，姓名是谁？"
        "项目经历中谁负责过项目的调研、设计、实现、测试、维护等全生命周期事宜？"
        "**请注意不需要标注出问题属于哪些**，例如**个人职业规划**"
    )
    # 在
    # /llama_index/core/evaluation/dataset_generation.py:246
    # /llama_index/core/evaluation/dataset_generation.py:274
    #  增加延时函数
    #  await asyncio.sleep(0.5)
    dataset_generator = DatasetGenerator.from_documents(
        docs,
        question_gen_query=base_question_gen_query,
        llm=llm,
        show_progress=True,
    )

    questions = dataset_generator.generate_questions_from_nodes(num=num)
    return questions


def gen_question_variations(generator_questions, llm: LLM, num_vary=3) -> list[str]:
    vary_question_tmpl = """\
你是一位公司HR。给定一个关于招聘的问题，你的目标是生成多达 {num_vary} 个问题变体，涉及多个求职者简历。

这可能包括比较/对比不同求职者简历，用另一个简历替换当前的简历，或生成只能通过多个关于简历回答的问题（发挥创意！）

你被提供了一组有效的求职者简历。请仅生成可以在该组简历中回答的问题变体。

For example:
Base Question: "哪位求职者拥有最多的项目管理经验？"
Valid 10Qs: [《关于张三的简历信息》, 《关于李四的简历信息》, 《关于王五的简历信息》]
Question Variations:
在张三和李四之间，谁有更多的领导经验？
如果我想找一个有强大数据分析技能的求职者，我应该选择哪个简历？
对比王五和张三的简历，哪个更适合市场营销职位？
基于现有简历，谁显示出最强的团队合作能力？

现在让我们试试吧！

Base Question: {base_question}
Valid 10Qs: {valid_10qs}
Question Variations:
"""

    VALID_10Q_STR = "[关于张毛峰的简历信息, 关于刘立兼的简历信息, 关于宋金珂的简历信息]"

    prompt_tmpl = PromptTemplate(vary_question_tmpl)

    new_questions = []
    for idx, question in enumerate(generator_questions):
        new_questions.append(question)
        response = llm.complete(
            prompt_tmpl.format(
                num_vary=num_vary,
                base_question=question,
                valid_10qs=VALID_10Q_STR,
            )
        )
        # parse into newlines
        raw_lines = str(response).split("\n")
        cur_new_questions = [l for l in raw_lines if l != ""]
        print(f"[{idx}] Original Question: {question}")
        print(f"[{idx}] Generated Question Variations: {cur_new_questions}")
        new_questions.extend(cur_new_questions)

    return new_questions
