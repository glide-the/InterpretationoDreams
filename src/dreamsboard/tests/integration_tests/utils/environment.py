"""Integration Test Environment:
Requires external API access.
Set the following environment variables before running:
  - OPENAI_API_KEY
  - CROSS_ENCODER_PATH
  - EMBED_MODEL_PATH
Optionally configure ChatOpenAI profiles via ``CHAT_OPENAI_PROFILES_PATH``
or ``CHAT_OPENAI_PROFILES_JSON``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

CHAT_OPENAI_IMPORT_SKIP = "Skipping test: ChatOpenAI dependency is not installed."

REQUIRED_ENV_VARS: tuple[str, ...] = ("OPENAI_API_KEY",)
OPTIONAL_ENV_VARS: tuple[str, ...] = (
    "CROSS_ENCODER_PATH",
    "EMBED_MODEL_PATH",
)
ALL_ENV_VARS: tuple[str, ...] = REQUIRED_ENV_VARS + OPTIONAL_ENV_VARS
SKIP_MESSAGE = "Skipping test: Missing external API credentials."
OPTIONAL_SKIP_TEMPLATE = (
    "Skipping test: {name} is required for this integration test."
)
START_TASK_CONTEXT: Path = (
    Path(__file__).resolve().parents[1] / "fixtures" / "task_context.json"
)
DEFAULT_CHAT_MODEL = os.getenv("CHAT_OPENAI_MODEL", "gpt-4o-mini")
CHAT_OPENAI_DEFAULTS: Dict[str, Any] = {
    "model": DEFAULT_CHAT_MODEL,
    "temperature": float(os.getenv("CHAT_OPENAI_TEMPERATURE", "0")),
    "max_retries": int(os.getenv("CHAT_OPENAI_MAX_RETRIES", "2")),
    "request_timeout": int(
        os.getenv("CHAT_OPENAI_TIMEOUT", "120")
    ),
}

DEFAULT_CHAT_OPENAI_PROFILES_PATH: Path = (
    Path(__file__).resolve().parents[1] / "config" / "chat_openai_profiles.json"
)
CHAT_OPENAI_PROFILES_ENV = "CHAT_OPENAI_PROFILES_JSON"
CHAT_OPENAI_PROFILES_PATH_ENV = "CHAT_OPENAI_PROFILES_PATH"
ENV_VALUE_PREFIX = "env:"
CHAT_OPENAI_PROFILE_SKIP_TEMPLATE = (
    "Skipping test: ChatOpenAI profile '{profile}' requires '{variable}'."
)


def missing_required_env_vars(env: Iterable[str] | None = None) -> List[str]:
    """Return the required environment variables that are not set."""

    env = env or REQUIRED_ENV_VARS
    return [var for var in env if not os.getenv(var)]


def ensure_external_services_available() -> None:
    """Skip the current test when required credentials are missing."""

    missing = missing_required_env_vars()
    if missing:
        pytest.skip(SKIP_MESSAGE)


def get_env_path(name: str) -> Optional[Path]:
    """Return a :class:`~pathlib.Path` for ``name`` when it is configured."""

    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def require_env_path(name: str) -> Path:
    """Return the resolved path for ``name`` or skip the current test."""

    path = get_env_path(name)
    if path is None:
        pytest.skip(OPTIONAL_SKIP_TEMPLATE.format(name=name))
    if not path.exists():
        pytest.skip(
            OPTIONAL_SKIP_TEMPLATE.format(
                name=f"{name} (missing resource at {path})"
            )
        )
    return path


def load_start_task_context(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the shared start task context JSON payload."""

    context_path = Path(path or START_TASK_CONTEXT)
    if not context_path.exists():
        raise FileNotFoundError(
            f"Unable to locate shared task context file: {context_path}"
        )
    with context_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _import_chat_openai() -> Type[Any]:
    """Return the available ``ChatOpenAI`` implementation or skip the test."""

    try:
        from langchain_openai import ChatOpenAI as Impl  # type: ignore

        return Impl
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOpenAI as Impl  # type: ignore

            return Impl
        except ImportError:
            pytest.skip(CHAT_OPENAI_IMPORT_SKIP)


def resolve_chat_openai_class() -> type[Any]:
    """Return the available ``ChatOpenAI`` implementation or skip the test."""

    return _import_chat_openai()


def _load_profiles_from_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _raw_chat_openai_profiles() -> Dict[str, Any]:
    raw: Dict[str, Any] = {}

    env_json = os.getenv(CHAT_OPENAI_PROFILES_ENV)
    if env_json:
        try:
            raw.update(json.loads(env_json))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                "Invalid JSON provided via"
                f" {CHAT_OPENAI_PROFILES_ENV}: {exc}"  # noqa: EM102
            ) from exc

    env_path = os.getenv(CHAT_OPENAI_PROFILES_PATH_ENV)
    if env_path:
        raw.update(_load_profiles_from_path(Path(env_path).expanduser().resolve()))

    if DEFAULT_CHAT_OPENAI_PROFILES_PATH.exists():
        raw.update(_load_profiles_from_path(DEFAULT_CHAT_OPENAI_PROFILES_PATH))

    profiles = raw.get("profiles", raw)
    if not isinstance(profiles, dict):  # pragma: no cover - defensive branch
        raise ValueError("ChatOpenAI profiles must be a mapping of names to configs.")
    return profiles


def _resolve_profile_value(
    profile: str, key: str, value: Any, skip_message: str
) -> Any:
    if isinstance(value, str) and value.startswith(ENV_VALUE_PREFIX):
        spec = value[len(ENV_VALUE_PREFIX) :]
        variable, _, fallback = spec.partition("||")
        env_value = os.getenv(variable)
        if env_value is None:
            if fallback:
                if fallback.startswith(ENV_VALUE_PREFIX):
                    return _resolve_profile_value(profile, key, fallback, skip_message)
                return fallback
            pytest.skip(
                CHAT_OPENAI_PROFILE_SKIP_TEMPLATE.format(
                    profile=profile, variable=variable
                )
            )
        return env_value
    if value == "<skip>":
        pytest.skip(skip_message)
    return value


def _build_chat_openai_profiles() -> Dict[str, Dict[str, Any]]:
    raw_profiles = _raw_chat_openai_profiles()
    built: Dict[str, Dict[str, Any]] = {}

    def _build(profile: str, stack: Optional[List[str]] = None) -> Dict[str, Any]:
        if profile in built:
            return built[profile]

        stack = stack or []
        if profile in stack:  # pragma: no cover - defensive branch
            raise ValueError(f"Circular ChatOpenAI profile inheritance: {stack + [profile]}")

        entry = raw_profiles.get(profile, {})
        inherit = entry.get("inherit") if isinstance(entry, dict) else None
        base = CHAT_OPENAI_DEFAULTS.copy()
        if inherit:
            base.update(_build(inherit, stack + [profile]))

        if isinstance(entry, dict):
            params = entry.get("parameters", {})
            if not params:
                params = {key: value for key, value in entry.items() if key != "inherit"}
        else:  # pragma: no cover - defensive branch
            params = {}

        resolved = {
            key: _resolve_profile_value(profile, key, value, SKIP_MESSAGE)
            for key, value in params.items()
        }
        merged = {key: value for key, value in {**base, **resolved}.items() if value is not None}
        built[profile] = merged
        return merged

    profiles_to_build = set(raw_profiles) | {"default"}
    for profile_name in profiles_to_build:
        _build(profile_name)

    return built


CHAT_OPENAI_PROFILES: Dict[str, Dict[str, Any]] = _build_chat_openai_profiles()


def get_chat_openai_profile(profile: str = "default") -> Dict[str, Any]:
    """Return the resolved ChatOpenAI parameter set for ``profile``."""

    if profile not in CHAT_OPENAI_PROFILES:
        available = ", ".join(sorted(CHAT_OPENAI_PROFILES))
        raise KeyError(
            f"Unknown ChatOpenAI profile '{profile}'. Available profiles: {available}."
        )
    return CHAT_OPENAI_PROFILES[profile].copy()


def create_chat_openai(profile: str = "default", **overrides: Any) -> Any:
    """Instantiate a ChatOpenAI-compatible object using an environment profile."""

    ensure_external_services_available()
    params = get_chat_openai_profile(profile)
    params.update(overrides)
    params = {key: value for key, value in params.items() if value is not None}

    chat_cls = resolve_chat_openai_class()
    return chat_cls(**params)


__all__ = [
    "ALL_ENV_VARS",
    "CHAT_OPENAI_DEFAULTS",
    "CHAT_OPENAI_PROFILES",
    "CHAT_OPENAI_IMPORT_SKIP",
    "OPTIONAL_SKIP_TEMPLATE",
    "REQUIRED_ENV_VARS",
    "SKIP_MESSAGE",
    "START_TASK_CONTEXT",
    "create_chat_openai",
    "ensure_external_services_available",
    "get_chat_openai_profile",
    "get_env_path",
    "resolve_chat_openai_class",
    "require_env_path",
    "load_start_task_context",
    "missing_required_env_vars",
]
