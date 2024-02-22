import os

import pytest
import logging
from dreamsboard.utils import get_config_dict, get_log_file, get_timestamp_ms


@pytest.fixture
def setup_log():
    logging_conf = get_config_dict(
        "DEBUG",
        get_log_file(log_path="logs", sub_dir=f"local_{get_timestamp_ms()}"),
        122,
        111,
    )

    logging.config.dictConfig(logging_conf)  # type: ignore
    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

    # wandb documentation to configure wandb using env variables
    # https://docs.wandb.ai/guides/track/advanced/environment-variables
    # here we are configuring the wandb project name
    os.environ["WANDB_PROJECT"] = "msg_extract_storage_2024_02_22"
    os.environ["WANDB_API_KEY"] = "local-08fe972f17575883ab88ed3cd49da17a5fc85f6d"
    os.environ["WANDB_BASE_URL"] = "http://localhost:8080"
