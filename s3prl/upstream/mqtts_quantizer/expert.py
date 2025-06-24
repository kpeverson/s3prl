import json

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .mqtts_models import *
from ..interfaces import UpstreamBase

class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, model_config: str = None, **kwargs):
        super().__init__(**kwargs)
        
        with open(model_config) as f:
            h = f.read()
        h = json.loads(h)
        h = AttrDict(h)
        h = infer_model_architecture(h)
        self.downsample_rate = h.sample2feature_ratio

        print(f"Loading model class {h.model_class}")
        model_cls = eval(h.model_class)
        self.model = model_cls(h, ckpt)
        # self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    def get_downsample_rates(self, feature_selection: str) -> int:
        return self.downsample_rate
    
    def forward(self, wavs):
        # print(f"wavs ({type(wavs)}, len {len(wavs)}): {[w.shape for w in wavs]}")
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)
        feats = self.model(padded_wav)
        # print(f"feats: {feats.shape}")

        return {"features": feats}