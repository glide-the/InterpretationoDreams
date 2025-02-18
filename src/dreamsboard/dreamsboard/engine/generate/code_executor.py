from typing import List

from langchain.schema import (
    AIMessage,
    SystemMessage,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.runnables import Runnable


# 创建一个代码执行器
class CodeExecutor:
    executor_code: str
    _messages: List[BaseMessage] = []
    _llm_runable: Runnable[LanguageModelInput, BaseMessage]
    _ai_message: AIMessage

    def __init__(
        self,
        executor_code: str,
        llm_runable: Runnable[LanguageModelInput, BaseMessage],
        messages: List[BaseMessage] = [],
    ):
        self.executor_code = executor_code
        self._llm_runable = llm_runable
        self._messages = messages

    def execute(self):
        # 创建变量字典
        variables = {
            "messages": self._messages,  # type: list[BaseMessage]
            "llm_runable": self._llm_runable,  # type: Runnable[LanguageModelInput, BaseMessage]
        }

        exec(self.executor_code, variables)

        self._llm_runable = variables.get("llm_runable", None)
        self._messages = variables.get("messages", None)

    def chat_run(self) -> AIMessage:
        # 创建变量字典
        variables = {
            "messages": self._messages,
            "llm_runable": self._llm_runable,
            "ai_message": None,
        }

        exec_code = "ai_message = llm_runable.invoke(messages)\r\n"
        exec(exec_code, variables)
        self._ai_message = variables.get("ai_message", None)
        return self._ai_message
