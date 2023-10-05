from typing import Dict, List, Optional, Union, Any
from typing_extensions import Self
from abc import abstractmethod
from pydantic import BaseModel, Field
from enum import Enum, auto
from jinja2 import Template
import uuid
import json
import textwrap
import hashlib
import logging
logger = logging.getLogger(__name__)
# NOTE: for pretty printing
TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class BaseComponent(BaseModel):
    """Base component object to caputure class names."""

    @classmethod
    @abstractmethod
    def class_name(cls) -> str:
        """Get class name."""

    def to_dict(self, **kwargs: Any) -> Dict[str, Any]:
        data = self.dict(**kwargs)
        data["class_name"] = self.class_name()
        return data

    def to_json(self, **kwargs: Any) -> str:
        data = self.to_dict(**kwargs)
        return json.dumps(data)

    # TODO: return type here not supported by current mypy version
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)
        return cls(**data)

    @classmethod
    def from_json(cls, data_str: str, **kwargs: Any) -> Self:  # type: ignore
        data = json.loads(data_str)
        return cls.from_dict(data, **kwargs)


class NodeRelationship(str, Enum):
    """Node relationships used in `BaseNode` class.

    Attributes:
        SOURCE: The node is the source document.
        # PREVIOUS: The node is the previous node in the document.
        # NEXT: The node is the next node in the document.
        # PARENT: The node is the parent node in the document.
        # CHILD: The node is a child node in the document.

    """

    SOURCE = auto()
    # PREVIOUS = auto()
    # NEXT = auto()
    # PARENT = auto()
    # CHILD = auto()


class ObjectTemplateType(str, Enum):
    BaseProgramGenerator = auto()
    QueryProgramGenerator = auto()
    AIProgramGenerator = auto()
    EngineProgramGenerator = auto()


class RelatedNodeInfo(BaseComponent):
    node_id: str
    node_type: Optional[ObjectTemplateType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    hash: Optional[str] = None

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "RelatedNodeInfo"


RelatedNodeType = Union[RelatedNodeInfo, List[RelatedNodeInfo]]


# Node classes for indexes
class BaseNode(BaseComponent):
    """Base node Object.

    Generic abstract interface for retrievable nodes

    """

    class Config:
        allow_population_by_field_name = True

    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique ID of the node."
    )

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - injected as part of the text for generating embeddings
    - used by vector DBs for metadata filtering

    """
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flat dictionary of metadata fields",
        alias="extra_info",
    )

    relationships: Dict[NodeRelationship, RelatedNodeType] = Field(
        default_factory=dict,
        description="A mapping of relationships to other node information.",
    )

    hash: str = Field(default="", description="Hash of the node content.")

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Object type."""

    @property
    @abstractmethod
    def template_content(self) -> str:
        """Get base_template_content."""

    @template_content.setter
    @abstractmethod
    def template_content(self, _template_content) -> None:
        """Get base_template_content."""

    @property
    @abstractmethod
    def render_data(self) -> dict:
        """Get render_data."""

    @render_data.setter
    @abstractmethod
    def render_data(self, _render_data: dict) -> None:
        """Get render_data."""

    @property
    @abstractmethod
    def render_code(self) -> str:
        """Get render_code."""

    @render_code.setter
    @abstractmethod
    def render_code(self, _exec_code: str) -> None:
        """Get render_code."""

    @property
    def node_id(self) -> str:
        return self.id_

    @node_id.setter
    def node_id(self, value: str) -> None:
        self.id_ = value

    @property
    def source_node(self) -> Optional[RelatedNodeInfo]:
        """Source object node.

        Extracted from the relationships field.

        """
        if NodeRelationship.SOURCE not in self.relationships:
            return None

        relation = self.relationships[NodeRelationship.SOURCE]
        if isinstance(relation, list):
            raise ValueError("Source object must be a single RelatedNodeInfo object")
        return relation

    @property
    def ref_template_id(self) -> Optional[str]:
        """Deprecated: Get ref doc id."""
        source_node = self.source_node
        if source_node is None:
            return None
        return source_node.node_id

    @property
    def extra_info(self) -> Dict[str, Any]:
        """TODO: DEPRECATED: Extra info."""
        return self.metadata

    def as_related_node_info(self) -> RelatedNodeInfo:
        """Get node as RelatedNodeInfo."""
        node_type = None
        try:
            node_type = ObjectTemplateType[self.get_type()]
            logger.info(f"The corresponding enum value for '{self.get_type()}' is: {node_type}")
        except KeyError:
            logger.info(f"'{self.get_type()}' is not a valid enum value.")
        return RelatedNodeInfo(
            node_id=self.node_id, node_type=node_type, metadata=self.metadata, hash=self.hash
        )

    def generate(self, render_data: dict = {}) -> str:
        logger.info(f'{self.__class__},生成代码')
        base_template = Template(self.template_content)
        if render_data is not None and self.render_data is not None:
            self.render_data = {**render_data, **self.render_data}
        else:
            # 处理其中一个或两者都为 None 的情况
            self.render_data = render_data or self.render_data or {}

        self.render_code = base_template.render(self.render_data)

        logger.info(f'{self.__class__},生成代码成功 {self.calculate_md5()}')
        return self.render_code

    def calculate_md5(self):
        md5_hash = hashlib.md5()
        md5_hash.update(self.render_code.encode('utf-8'))
        return md5_hash.hexdigest()

    def __str__(self) -> str:
        source_text_truncated = truncate_text(
            self.render_code.strip(), TRUNCATE_LENGTH
        )
        source_text_wrapped = textwrap.fill(
            f"Text: {source_text_truncated}\n", width=WRAP_WIDTH
        )
        return f"Node ID: {self.node_id}\n{source_text_wrapped}"

