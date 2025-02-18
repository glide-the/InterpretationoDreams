import os
import time

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.openai import (
    OpenAIEmbedding,
)
from llama_index.finetuning.callbacks import OpenAIFineTuningHandler

# pants: no-infer-dep
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import GPT4_MODELS

from dreamsboard.agent.react_agent_build_query_engine_tools import (
    build_query_docs,
    build_query_docs_index_store,
    build_query_engine_tools,
)
from dreamsboard.agent.react_tools_utils import load_questions, save_questions

if __name__ == "__main__":
    llm = OpenAI(
        model="glm-4",
        temperature=0.99,
        api_key="a9733a59370d34ef51d261bd461251a5.YD9PU7sAqUwgD8Sk",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
    )
    embeddings = OpenAIEmbedding(
        api_key="EMPTY",
        model="text-embedding-3-large",
        api_base="http://127.0.0.1:9997/v1",
    )
    query_docs = build_query_docs()
    index_store = build_query_docs_index_store(query_docs, embeddings)
    query_engine_tools = build_query_engine_tools(llm, index_store)
    # 获取当前文件路径
    root_dir = os.path.dirname(os.path.abspath(__file__))
    train_questions = load_questions(
        os.path.join(root_dir, f"glm_train_questions_1152q.txt")
    )
    eval_questions = load_questions(
        os.path.join(root_dir, f"glm_eval_questions_288q.txt")
    )
    print(len(eval_questions))

    finetuning_handler = OpenAIFineTuningHandler()
    callback_manager = CallbackManager([finetuning_handler])
    # limit the context window artifically to test refine process
    Settings.context_window = 4096
    agent_llm = OpenAI(
        model="glm-4",
        temperature=0.01,
        api_key="a9733a59370d34ef51d261bd461251a5.YD9PU7sAqUwgD8Sk",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
    )
    gpt4_agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=agent_llm,
        callback_manager=callback_manager,
        verbose=True,
    )
    # 增加一个时间戳
    # train_questions = [f"{q} {time.time()}" for q in train_questions]
    # 每10次迭代保存一次模型
    for idx, question in enumerate(train_questions):
        try:
            print(f"[{idx}] Question: {question}")
            response = gpt4_agent.query(question)
            print(f"[{idx}] Agent Response: {str(response)}")
            if idx % 10 == 0:
                # save events
                print(f"[{idx}] Saving finetuning events...")
                finetuning_handler.save_finetuning_events(
                    os.path.join(
                        root_dir,
                        f"glm_q_gemini_a_finetuning_events_{idx}q_{time.time()}.jsonl",
                    )
                )
        except Exception as e:
            print(f"[{idx}] Error: {e}")
    # save events
    finetuning_handler.save_finetuning_events(
        os.path.join(
            root_dir,
            f"glm_q_gemini_a_finetuning_events_{len(train_questions)}q_{time.time()}.jsonl",
        )
    )
