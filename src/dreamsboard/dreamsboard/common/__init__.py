from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from dreamsboard.common.struct_type import AdapterAllToolStructType


def _is_assistants_builtin_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> bool:
    """platform tools built-in"""
    assistants_builtin_tools = AdapterAllToolStructType.__members__.values()
    return (
        isinstance(tool, dict)
        and ("type" in tool)
        and (tool["type"] in assistants_builtin_tools)
    )


def _get_assistants_tool(
    tool: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    """Convert a raw function/class to an ZhipuAI tool."""
    if _is_assistants_builtin_tool(tool):
        return tool  # type: ignore
    else:
        # in case of a custom tool, convert it to an function of type
        return convert_to_openai_tool(tool)
