# -*- coding: utf-8 -*-
"""AdapterAllToolStructType class."""

from enum import Enum


class AdapterAllToolStructType(str, Enum):
    """

    Attributes:
        DICT ("dict"):

    """

    CODE_INTERPRETER = "code_interpreter"
    DRAWING_TOOL = "drawing_tool"
    WEB_BROWSER = "web_browser"
    WEB_SEARCH = "web_search"
