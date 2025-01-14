from langchain_openai import ChatOpenAI

from dreamsboard.document_loaders import KorLoader
from dreamsboard.document_loaders.protocol.ner_protocol import DreamsStepInfo
from dreamsboard.engine.storage.dreams_analysis_store.simple_dreams_analysis_store import SimpleDreamsAnalysisStore


def test_kor_glm_3():
    dreams_analysis_store = SimpleDreamsAnalysisStore.from_persist_dir(persist_dir="./storage")
    dreams_guidance_context = None
    dreams_personality_context = None
    for val in dreams_analysis_store.analysis_all.values():
        dreams_guidance_context = val.dreams_guidance_context
        dreams_personality_context = val.dreams_personality_context
    guidance_llm = ChatOpenAI(
        openai_api_base='http://0.0.0.0:8000/v1',
        model="glm-4",
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
    kor_dreams_guidance_chain = KorLoader.form_kor_dreams_guidance_builder(
        llm_runable=guidance_llm)

    response = kor_dreams_guidance_chain.run(dreams_guidance_context)
    dreams_step_list = []
    if response.get('data') is not None and response.get('data').get('script') is not None:
        step_list = response.get('data').get('script')
        for step in step_list:
            dreams_step = DreamsStepInfo(step_advice=step.get('step_advice'),
                                         step_description=step.get('step_description'))
            dreams_step_list.append(dreams_step)

    print(dreams_step_list)


def test_kor_glm_4():
    dreams_analysis_store = SimpleDreamsAnalysisStore.from_persist_dir(persist_dir="./storage")
    dreams_guidance_context = None
    dreams_personality_context = None
    for val in dreams_analysis_store.analysis_all.values():
        dreams_guidance_context = val.dreams_guidance_context
        dreams_personality_context = val.dreams_personality_context
    guidance_llm = ChatOpenAI(
        openai_api_base='https://open.bigmodel.cn/api/paas/v4/',
        model="glm-4",
        verbose=True,
        temperature=0.95,
        top_p=0.70,
    )
    kor_dreams_guidance_chain = KorLoader.form_kor_dreams_guidance_builder(
        llm_runable=guidance_llm)

    response = kor_dreams_guidance_chain.run(dreams_guidance_context)
    dreams_step_list = []
    if response.get('data') is not None and response.get('data').get('script') is not None:
        step_list = response.get('data').get('script')
        for step in step_list:
            dreams_step = DreamsStepInfo(step_advice=step.get('step_advice'),
                                         step_description=step.get('step_description'))
            dreams_step_list.append(dreams_step)

    print(dreams_step_list)
