from llama_index.core import Document, PromptTemplate
from llama_index.core.agent import ReActAgent
from llama_index.core.evaluation import DatasetGenerator
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI


def adapter_agent(llm_agent: OpenAI, query_engine_tools):
    base_agent = ReActAgent.from_tools(query_engine_tools, llm=llm_agent, verbose=True)
    return base_agent


def build_dataset_generator_questions(
    base_question_gen_query, docs: list[Document], llm: LLM, num=60
) -> list[str]:
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


def gen_question_variations(
    vary_question_tmpl, valid_10q_str, generator_questions, llm: LLM, num_vary=3
) -> list[str]:
    prompt_tmpl = PromptTemplate(vary_question_tmpl)

    new_questions = []
    for idx, question in enumerate(generator_questions):
        new_questions.append(question)
        response = llm.complete(
            prompt_tmpl.format(
                num_vary=num_vary,
                base_question=question,
                valid_10qs=valid_10q_str,
            )
        )
        # parse into newlines
        raw_lines = str(response).split("\n")
        cur_new_questions = [l for l in raw_lines if l != ""]
        print(f"[{idx}] Original Question: {question}")
        print(f"[{idx}] Generated Question Variations: {cur_new_questions}")
        new_questions.extend(cur_new_questions)

    return new_questions
