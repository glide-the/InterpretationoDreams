from __future__ import annotations
"""Integration test fixtures for `dreamsboard` suites.

The integration tests require external API access. Before running ensure the
following environment variables are configured:

* ``OPENAI_API_KEY``
* ``CROSS_ENCODER_PATH``
* ``EMBED_MODEL_PATH``
"""

from typing import Any, Callable, Dict

import pytest

from .utils.environment import (
    START_TASK_CONTEXT,
    ensure_external_services_available,
    get_chat_openai_profile,
    load_start_task_context,
    resolve_chat_openai_class,
    require_env_path,
)


@pytest.fixture(scope="session", autouse=True)
def integration_environment() -> None:
    """Validate integration environment variables before running tests."""

    ensure_external_services_available()


@pytest.fixture(scope="session")
def chat_openai_defaults() -> Dict[str, Any]:
    """Expose the canonical ChatOpenAI configuration for the tests."""

    return get_chat_openai_profile()


@pytest.fixture(scope="session")
def start_task_context_payload() -> Dict[str, Any]:
    """Load the shared start task context file used by integration tests."""

    return load_start_task_context()


@pytest.fixture(scope="session")
def start_task_context_path() -> str:
    """Return the path to the shared task context JSON file."""

    return str(START_TASK_CONTEXT)


@pytest.fixture(scope="session")
def start_task_context(start_task_context_payload: Dict[str, Any]) -> str:
    """Expose the canonical start task context string."""

    return start_task_context_payload["start_task_context"]


@pytest.fixture(scope="session")
def chat_openai_class(integration_environment: None):
    """Resolve the ChatOpenAI-compatible class for integration tests."""

    chat_class = resolve_chat_openai_class()

    try:  # pragma: no cover - optional dependency alignment
        import langchain_community.chat_models as community_models

        community_models.ChatOpenAI = chat_class  # type: ignore[attr-defined]
    except ImportError:
        pass

    try:  # pragma: no cover - optional dependency alignment
        import langchain_openai as openai_module

        openai_module.ChatOpenAI = chat_class  # type: ignore[attr-defined]
    except ImportError:
        pass

    return chat_class


@pytest.fixture(scope="session")
def chat_openai_factory(
    chat_openai_class,
    chat_openai_defaults: Dict[str, Any],
) -> Callable[..., Any]:
    """Return a factory that instantiates ChatOpenAI-compatible clients."""

    def _factory(profile: str = "default", **overrides: Any) -> Any:
        if profile == "default":
            params = {**chat_openai_defaults}
        else:
            params = get_chat_openai_profile(profile)
        params.update(overrides)
        params = {key: value for key, value in params.items() if value is not None}
        return chat_openai_class(**params)

    return _factory


@pytest.fixture(scope="session")
def cross_encoder_path() -> str:
    """Return the configured cross-encoder model path or skip if missing."""

    return str(require_env_path("CROSS_ENCODER_PATH"))


@pytest.fixture(scope="session")
def embed_model_path() -> str:
    """Return the configured embedding model path or skip if missing."""

    return str(require_env_path("EMBED_MODEL_PATH"))
