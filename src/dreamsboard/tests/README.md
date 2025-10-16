# Dreamsboard Integration Test Guide

This guide documents the environment requirements for `src/dreamsboard/tests/integration_tests` so that contributors can reliably run the suite locally and in CI. The structure mirrors the [LangChain testing requirements](https://docs.langchain.com/oss/python/contributing/code#testing-requirements) by clearly separating external dependencies, configuration, and skip logic.

## 1. 外部服务 / API 要求

| 服务 | 使用位置 | 关联环境变量 | 说明与策略 |
| --- | --- | --- | --- |
| OpenAI 兼容对话模型 | `test_aemo_representation_chain` 链构建与提示调用 | `OPENAI_API_KEY` (必需) | 若未设置则集成测试整体跳过，同时给出缺失凭据的提示信息。 |
| Cross-Encoder 重排序模型 | 语义检索/重排序相关链 | `CROSS_ENCODER_PATH` (可选) | 支持本地权重路径或 Hugging Face 标识。缺失时依赖 `cross_encoder_path` fixture 的测试会自动 `skip`。 |
| Embedding 模型/向量索引 | 向量检索构建流程 | `EMBED_MODEL_PATH` (可选) | 指向 FAISS/向量模型存储目录。缺失时依赖 `embed_model_path` fixture 的测试会自动 `skip`。 |

若未设置必需的外部服务凭据，`pytest` 会输出 `Skipping test: Missing external API credentials.` 并自动跳过对应用例。

## 2. 配置项说明

- `START_TASK_CONTEXT`: 统一的任务上下文路径，位于 `src/dreamsboard/tests/integration_tests/fixtures/task_context.json`，通过 `load_start_task_context()` 读取。
- `CHAT_OPENAI_DEFAULTS`: ChatOpenAI 客户端统一默认配置（模型、温度、超时等），详见 `src/dreamsboard/tests/integration_tests/utils/environment.py`。
- `config/chat_openai_profiles.json`: 通过集中式配置管理所有 ChatOpenAI 运行参数，避免在测试中硬编码模型地址或网关。若需要自定义，可设置 `CHAT_OPENAI_PROFILES_PATH` 指向自定义 JSON，或通过 `CHAT_OPENAI_PROFILES_JSON` 传入 JSON 字符串。
- `CHAT_OPENAI_MODEL`、`CHAT_OPENAI_TEMPERATURE`、`CHAT_OPENAI_MAX_RETRIES`、`CHAT_OPENAI_TIMEOUT`：用于覆盖默认的对话模型、温度、重试次数与请求超时时间，便于在不同环境下统一配置。
- `get_env_path(name)`: 帮助函数，用于解析模型路径环境变量（如 `CROSS_ENCODER_PATH`）。
- `require_env_path(name)`: 若对应环境变量缺失则通过 `pytest.skip` 中止当前测试，保证测试人员能够清晰获知缺失的资源。

## 3. ChatOpenAI 使用规范

- **版本与导入**：默认使用 `langchain_openai.ChatOpenAI` 或 `langchain_community.chat_models.ChatOpenAI`。
- **默认配置**：`CHAT_OPENAI_DEFAULTS` 中定义，模型默认为 `gpt-4o-mini`，`temperature=0`，`max_retries=2`，`request_timeout=120s`。
- **环境 Profile**：使用 `create_chat_openai(profile="<name>")` 调用 `config/chat_openai_profiles.json` 中定义的参数集合，例如：

  ```python
  from dreamsboard.tests.integration_tests.utils.environment import create_chat_openai

  llm = create_chat_openai(profile="glm4_plus_low_temp")
  guidance_llm = create_chat_openai(profile="glm4_plus_guidance")
  ```

  Profile 支持环境变量占位符（如 `env:OPENAI_API_BASE` 或带回退的 `env:DEEPSEEK_API_BASE||env:OPENAI_API_BASE`），缺失时会 `pytest.skip` 提示所需变量。
- **配置来源**：所有参数来自统一的 JSON 配置文件（默认 `config/chat_openai_profiles.json`）。该文件可在本地覆写，或通过 `CHAT_OPENAI_PROFILES_JSON` 环境变量直接提供 JSON 内容。

## 4. 辅助工具与共享配置

- `integration_environment` fixture：在 `conftest.py` 中自动运行，确保满足环境条件或在缺少凭据时跳过测试。
- `start_task_context_payload` fixture：加载统一的任务上下文 JSON。
- `cross_encoder_path` fixture：解析 `CROSS_ENCODER_PATH` 并在缺失时自动跳过当前测试。
- `embed_model_path` fixture：解析 `EMBED_MODEL_PATH` 并在缺失时自动跳过当前测试。
- 所有工具函数位于 `src/dreamsboard/tests/integration_tests/utils/environment.py`，包含对缺失凭据的 Skip 逻辑 (`ensure_external_services_available`)。

## 5. 运行流程

### 本地环境

1. 复制并填充 `.env.example`：

   ```bash
   cp .env.example .env
   export $(grep -v '^#' .env | xargs)
   ```

2. 安装依赖并运行：

   ```bash
   poetry install
   poetry run pytest src/dreamsboard/tests/integration_tests -m "not slow"
   ```

### CI 环境

1. 在 CI Secrets 中配置 `OPENAI_API_KEY`、`CROSS_ENCODER_PATH`、`EMBED_MODEL_PATH`。
2. 在流水线脚本中加载 `.env.example`，将 Secrets 写入运行环境。
3. 运行 `poetry run pytest src/dreamsboard/tests/integration_tests`。
4. 若 Secrets 不可用，测试会自动 Skip 并给出缺失凭据提示。

## 6. 参考

- LangChain 官方测试规范：确保网络依赖在凭据缺失时自动跳过。
- `src/dreamsboard/tests/integration_tests/test_aemo_representation_chain/prompts.py`：在模块注释中记录依赖与配置策略。
- `src/dreamsboard/tests/integration_tests/conftest.py`：集中处理 pytest 配置与环境校验逻辑。
- `src/dreamsboard/tests/integration_tests/utils/environment.py`：环境变量与共享工具的实现。
