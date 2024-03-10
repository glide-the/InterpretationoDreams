import os

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
)  # pants: no-infer-dep

from llama_index.llms.openai import OpenAI
from llama_index.finetuning.callbacks import OpenAIFineTuningHandler
from llama_index.core.callbacks import CallbackManager
from llama_index.core.agent import ReActAgent

from llama_index.core import Settings

from dreamsboard.agent.react_agent_build_query_engine_tools import build_query_docs, build_query_docs_index_store, \
    build_query_engine_tools
from dreamsboard.agent.react_tools_utils import save_questions, load_questions

if __name__ == '__main__':
    llm = OpenAI(model="gpt-4", temperature=0.3, api_key="sk-ApUK41y73g8qMbrz36A81641752946449f10BbBe32Ff2b7c",
                 api_base="http://localhost:3000/v1")
    embeddings = OpenAIEmbedding(api_key="EMPTY", api_base="http://127.0.0.1:9997/v1")
    query_docs = build_query_docs()
    index_store = build_query_docs_index_store(query_docs, embeddings)
    query_engine_tools = build_query_engine_tools(llm, index_store)
    # 获取当前文件路径
    root_dir = os.path.dirname(os.path.abspath(__file__))
    train_questions = load_questions(os.path.join(root_dir, f"train_questions_192q.txt"))
    eval_questions = load_questions(os.path.join(root_dir, f"eval_questions_48q.txt"))
    print(len(eval_questions))

    finetuning_handler = OpenAIFineTuningHandler()
    callback_manager = CallbackManager([finetuning_handler])
    # limit the context window artifically to test refine process
    Settings.context_window = 2048
    agent_llm = OpenAI(model="gpt-4", temperature=0.9, api_key="sk-ApUK41y73g8qMbrz36A81641752946449f10BbBe32Ff2b7c",
                       api_base="http://localhost:3000/v1")
    gpt4_agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=agent_llm,
        callback_manager=callback_manager,
        verbose=True,
    )
    # 每10次迭代保存一次模型
    for idx, question in enumerate(train_questions):
        try:
            print(f"[{idx}] Question: {question}")
            response = gpt4_agent.query(question)
            print(f"[{idx}] Agent Response: {str(response)}")
            if idx % 10 == 0:
                # save events
                print(f"[{idx}] Saving finetuning events...")
                finetuning_handler.save_finetuning_events(os.path.join(root_dir, f"finetuning_events_{idx}q.jsonl"))
        except Exception as e:
            print(f"[{idx}] Error: {e}")
    # save events
    finetuning_handler.save_finetuning_events(os.path.join(root_dir, f"finetuning_events_{len(train_questions)}q.jsonl"))
