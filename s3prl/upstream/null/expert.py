from collections import OrderedDict
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "[null UpstreamExpert]"

        print(
            f"Using null upstream model. Returns wav unchanged"
        )


    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 320
        """
        return 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        wavs = pad_sequence(wavs, batch_first=True).unsqueeze(-1)
        # wavs: (batch_size, max_len, 1)
        # downsample by 320
        wavs = wavs[:, ::320, :]

        return {
            "hidden_states": wavs,
        }
        # empty_tensor = torch.tensor([]).to(wavs[0].device)
        # wavs = [empty_tensor for _ in wavs]
        # return {
        #     "hidden_states": wavs,
        # }
