from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders import TextLoader


def test_loader_with_files():
    loader = DirectoryLoader('C:/Users/Administrator/Desktop/监管',
                             glob="**/*.md",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"},
                             use_multithreading=True,
                             show_progress=True)
    docs = loader.load()
