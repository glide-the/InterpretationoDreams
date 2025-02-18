import os

from llama_index.embeddings.openai import (
    OpenAIEmbedding,
)
from llama_index.llms.openai import OpenAI

# pants: no-infer-dep
from dreamsboard.agent.react_agent_build_query_engine_tools import (
    build_query_docs,
    build_query_docs_index_store,
    build_query_engine_tools,
)
from dreamsboard.agent.react_agent_finetune_cleaning import (
    adapter_agent,
    build_dataset_generator_questions,
    gen_question_variations,
)
from dreamsboard.agent.react_tools_utils import save_questions

if __name__ == "__main__":
    # 获取当前文件路径
    root_dir = os.path.dirname(os.path.abspath(__file__))
    llm = OpenAI(
        model="gpt-4",
        temperature=0.3,
        api_key="sk-ApUK41y73g8qMbrz36A81641752946449f10BbBe32Ff2b7c",
        api_base="http://localhost:3000/v1",
    )
    embeddings = OpenAIEmbedding(api_key="EMPTY", api_base="http://127.0.0.1:9997/v1")
    query_docs = build_query_docs()
    index_store = build_query_docs_index_store(query_docs, embeddings)
    query_engine_tools = build_query_engine_tools(llm, index_store)
    agent_llm = OpenAI(
        model="gpt-4",
        temperature=0.9,
        api_key="sk-ApUK41y73g8qMbrz36A81641752946449f10BbBe32Ff2b7c",
        api_base="http://localhost:3000/v1",
    )
    base_agent = adapter_agent(agent_llm, query_engine_tools)

    response = base_agent.chat("查询组织机构的接口是什么")
    print(str(response))
    base_question_gen_query = (
        "你是一名公司业务产品，你的任务是基于系统设计访问的思想设计一套电网拓扑、图形、资源、资产、测点等基础业务。"
        "使用业务接口文档提供的上下文， 制定一些问题，"
        "从上下文中捕捉到重要事实形成问题，"
        "将问题限制在所提供的上下文信息内."
        "**提取的事实需要验证,不需要标注出具体的来源和分类标签**"
        "例如："
        "  停/供电范围分析，在拓扑服务吗？"
        "  问题"
        "**请注意你只需要查看接口文档中的业务介绍、请求路径、请求方式，请忽略请求参数和示例信息、返回参数、调用范例**"
    )
    num = 360
    # 拼接 所有query_docs
    all_docs = []
    for doc in query_docs:
        all_docs.extend(doc)

    questions = build_dataset_generator_questions(
        base_question_gen_query=base_question_gen_query, docs=all_docs, llm=llm, num=num
    )
    print(len(questions))
    print(questions)
    save_questions(questions, os.path.join(root_dir, f"glm_questions_{num}q.txt"))
    vary_question_tmpl = """\
你是一位公司产品经理。给定一个关于接口文档的产品需求的问题，你的目标是生成多达 {num_vary} 个问题变体，涉及多个接口文档 。

这可能包括比较/对比不同接口文档，你可以通过业务介绍、请求路径、请求方式生成，或只能通过两个业务介绍的问题（发挥创意！）

你被提供了一组有效的接口文档。请仅生成可以在该组接口文档中回答的问题变体。

For example:
Base Question: 如何通过《OOS文件系统业务介绍》上传文件？
Valid 10Qs: [《组织人员业务接口》, 《任务调度系统接口》, 《OOS文件系统业务介绍》]
Question Variations: 
使用《任务调度系统接口》创建任务后，如何利用《OOS文件系统业务介绍》对任务结果进行存储？
在《组织人员业务接口》中添加新员工后，如何配置《OOS文件系统业务介绍》以分配个人文件存储空间？  
如何结合使用《OOS文件系统业务介绍》和《任务调度系统接口》来优化文件的自动备份流程？  

现在让我们试试吧！

Base Question: {base_question}
Valid 10Qs: {valid_10qs}
Question Variations:
"""

    VALID_10Q_STR = "[关于测点管理中心的业务中台, 关于电网拓扑中心的业务中台, 关于电网图形中心的业务中台, 关于电网资产中心的业务中台, 关于电网资源中心的业务中台, 关于电基础服务中心的业务中台]"

    new_questions = gen_question_variations(
        vary_question_tmpl=vary_question_tmpl,
        valid_10q_str=VALID_10Q_STR,
        generator_questions=questions,
        llm=llm,
        num_vary=3,
    )
    print(len(new_questions))
    print(new_questions)
    total_questions = num * (3 + 1)
    # 训练集和验证集，训练集占80%，验证集占20%
    train_questions, eval_questions = (
        new_questions[: int(total_questions * 0.8)],
        new_questions[int(total_questions * 0.8) :],
    )

    save_questions(
        train_questions,
        os.path.join(
            root_dir, f"glm_train_questions_{int(total_questions * 0.8)}q.txt"
        ),
    )
    save_questions(
        eval_questions,
        os.path.join(root_dir, f"glm_eval_questions_{int(total_questions * 0.2)}q.txt"),
    )
