
import logging
import os
import time
from pathlib import Path

from filelock import FileLock

from s3prl.util.download import _urls_to_filepaths
# import torch_tb_profiler

from .expert import UpstreamExpert as _UpstreamExpert

logger = logging.getLogger(__name__)

NEW_ENOUGH_SECS = 2.0

def mqtts_custom(
        ckpt: str = None,
        model_config: str = None,
        **kwargs,
):
    return _UpstreamExpert(ckpt=ckpt, model_config=model_config, **kwargs)