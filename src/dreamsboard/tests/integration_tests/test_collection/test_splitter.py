from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter


def test_splitter():
    loader = DirectoryLoader('C:/Users/Administrator/Desktop/监管',
                             glob="**/*.md",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"},
                             use_multithreading=True,
                             show_progress=True)
    files = loader.load()
    headers_to_split_on = [
        ("#", "head1"),
        ("##", "head2"),
        ("###", "head3"),
        ("####", "head4"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    all_chunks = []
    for docs in files:
        for doc in docs:

            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                if doc.metadata:
                    chunk.metadata["source"] = doc.metadata["source"]

            all_chunks.extend(chunks)

    print(all_chunks)