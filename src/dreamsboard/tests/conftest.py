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
    os.environ["WANDB_PROJECT"] = "dreams_wechat_batch_extract"
    os.environ["WANDB_API_KEY"] = "974207f7173417ef95d2ebad4cbe7f2f9668a093"
