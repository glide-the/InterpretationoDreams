import os

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
)  # pants: no-infer-dep

from dreamsboard.agent.react_agent_build_query_engine_tools import build_query_engine_tools, build_query_docs, \
    build_query_docs_index_store
from dreamsboard.agent.react_agent_finetune_cleaning import adapter_agent, build_dataset_generator_questions, \
    gen_question_variations
from dreamsboard.agent.react_tools_utils import save_questions

if __name__ == '__main__':
    llm = OpenAI(model="gpt-4", temperature=0.3, api_key="sk-ApUK41y73g8qMbrz36A81641752946449f10BbBe32Ff2b7c",
                 api_base="http://localhost:3000/v1")
    embeddings = OpenAIEmbedding(api_key="EMPTY", api_base="http://127.0.0.1:9997/v1")
    query_docs = build_query_docs()
    index_store = build_query_docs_index_store(query_docs, embeddings)
    query_engine_tools = build_query_engine_tools(llm, index_store)
    agent_llm = OpenAI(model="gpt-4", temperature=0.9, api_key="sk-ApUK41y73g8qMbrz36A81641752946449f10BbBe32Ff2b7c",
                       api_base="http://localhost:3000/v1")
    base_agent = adapter_agent(agent_llm, query_engine_tools)

    response = base_agent.chat(
        "有没有人做了硬件密钥项目，姓名是谁"
    )
    print(str(response))

    questions = build_dataset_generator_questions(query_docs[0], llm, num=60)
    print(len(questions))
    print(questions)

    new_questions = gen_question_variations(questions, llm, num_vary=3)
    print(len(new_questions))
    print(new_questions)
    total_questions = 60 * (3 + 1)
    # 训练集和验证集，训练集占80%，验证集占20%
    train_questions, eval_questions = new_questions[:int(total_questions * 0.8)], new_questions[
                                                                                  int(total_questions * 0.8):]
    # 获取当前文件路径
    root_dir = os.path.dirname(os.path.abspath(__file__))
    save_questions(train_questions, os.path.join(root_dir, f"gemini_train_questions_{int(total_questions * 0.8)}q.txt"))
    save_questions(eval_questions, os.path.join(root_dir, f"gemini_eval_questions_{int(total_questions * 0.2)}q.txt"))
