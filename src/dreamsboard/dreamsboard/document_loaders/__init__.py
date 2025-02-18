from dreamsboard.document_loaders.csv_structured_storyboard_loader import (
    StructuredStoryboardCSVBuilder,
)
from dreamsboard.document_loaders.kor_loader import KorLoader
from dreamsboard.document_loaders.structured_storyboard_loader import (
    LinkedListNode,
    StructuredStoryboard,
    StructuredStoryboardLoader,
)

__all__ = [
    "StructuredStoryboard",
    "LinkedListNode",
    "StructuredStoryboardLoader",
    "StructuredStoryboardCSVBuilder",
    "KorLoader",
]


def load_csv(file_dir):
    """
    获取输入文件夹内的所有txt文件，并返回文件名全称列表
    """
    import os

    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".csv":
                filename = os.path.join(root, file)
                L.append(filename)
        return L


def batch(iterable, size):
    # range对象的step是size
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]
