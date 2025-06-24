import sys
MQTTS_DIR = "/gscratch/tial/kpever/workspace/mqtts_training"
sys.path.append(MQTTS_DIR)

import librosa
import numpy as np
import scipy
import torch
import torch.nn as nn

from quantizer.models import Encoder, Quantizer, GlottalQuantizer

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def infer_model_architecture(h):
    """
    calculate model architecture from config
    """
    h.num_upsamples = len(h.upsample_rates)
    h.encoder_dim = h.encoder_input_dim*(2**h.num_upsamples)
    assert h.encoder_dim % h.n_code_groups == 0, f"encoder_dim {h.encoder_dim} must be divisible by n_code_groups {h.n_code_groups}"

    # calculate product of all upsample rates
    h.sample2feature_ratio = 1
    for rate in h.upsample_rates:
        h.sample2feature_ratio *= rate
    h.feature_rate = h.sampling_rate / h.sample2feature_ratio

    return h

class MQTTSEncoderQuantizer(torch.nn.Module):
    def __init__(self, h: AttrDict, ckpt: str, **kwargs):
        super().__init__()
        h = infer_model_architecture(h)
        self.encoder = Encoder(h)
        self.quantizer = Quantizer(h)

        self.load_checkpoint(ckpt)

        self.h = h

    def load_checkpoint(self, ckpt: str):
        ckpt = torch.load(ckpt, map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder"])
        self.quantizer.load_state_dict(ckpt["quantizer"])

    def forward(self, x):
        x = self.encoder(x.unsqueeze(1))
        x, _, _ = self.quantizer(x)
        return x.transpose(1, 2)
    
# class MQTTSParallelEncoderQuantizer(torch.nn.Module):
#     def __init__(self, h: AttrDict, **kwargs):
#         super().__init__()
#         h = infer_model_architecture(h)
#         self.encoder1 = Encoder(h)
#         self.encoder2 = Encoder(h)
#         self.quantizer1 = Quantizer(h)
#         self.quantizer2 = GlottalQuantizer(h)

#     def forward(self, x):
#         x1 = self.encoder1(x.unsqueeze(1))
#         x2 = self.encoder2(x.unsqueeze(1))

#         x1, _, _ = self.quantizer1(x1)
#         x2, _, _ = self.quantizer2(x2)

#         return torch.cat([x1, x2], dim=1)
    
class MQTTSParallelEncoderQuantizer(torch.nn.Module):
    def __init__(self, h: AttrDict, ckpt: str, **kwargs):
        super().__init__()
        h = infer_model_architecture(h)
        self.encoder1 = Encoder(h)
        self.encoder2 = Encoder(h)
        self.quantizer1 = Quantizer(h)
        self.quantizer2 = GlottalQuantizer(h)
        
        self.sr = h.sampling_rate
        self.audio2_mode = h.audio2_mode
        # self.audio2_mode = kwargs.get("audio2_mode", "copy")
        assert self.audio2_mode in ["copy", "glottal_lpc"], f"Unsupported audio2_mode: {self.audio2_mode}"
        print(f"audio2_mode: {self.audio2_mode}")

        if self.audio2_mode == "glottal_lpc":
            self.lpc_order = kwargs.get("lpc_order", 16)
            self.lpc_window = kwargs.get("lpc_window", "hamming")
            self.lpc_window_size = kwargs.get("lpc_window_size", 0.025)
            self.lpc_window_stride = kwargs.get("lpc_window_stride", 0.01)
            self.energy_threshold = kwargs.get("energy_threshold", 1e-4)
    
    def load_checkpoint(self, ckpt: str):
        ckpt = torch.load(ckpt, map_location="cpu")
        self.encoder1.load_state_dict(ckpt["encoder1"])
        self.encoder2.load_state_dict(ckpt["encoder2"])
        self.quantizer1.load_state_dict(ckpt["quantizer1"])
        self.quantizer2.load_state_dict(ckpt["quantizer2"])

    def forward_glottal(self, x):
        lpc_window_size = int(self.lpc_window_size * self.sr)
        lpc_window_stride = int(self.lpc_window_stride * self.sr)

        def inverse_filter(x_frame, a):
            if np.sum(x_frame**2) < self.energy_threshold:
                return x_frame
            x_frame_hat = scipy.signal.lfilter(
                np.hstack([[0], -1 * a[1:]]), [1], x_frame
            )
            return x_frame - x_frame_hat
        
        x = x.numpy()
        glottal_x = np.zeros_like(x)
        frames = librosa.util.frame(x, frame_length=lpc_window_size, hop_length=lpc_window_stride).T
        if self.lpc_window == "hamming":
            window = np.hamming(lpc_window_size)
        else:
            raise ValueError(f"Unsupported window type: {self.lpc_window}")
        
        for i, frame in enumerate(frames):
            frame = frame*window
            a = librosa.lpc(frame, self.lpc_order)
            frame_glottal_x = inverse_filter(frame, a)
            glottal_x[i*lpc_window_stride:i*lpc_window_stride+lpc_window_size] += frame_glottal_x

        return torch.tensor(glottal_x)

    def forward(self, x):
        if self.audio2_mode == "copy":
            x2 = x.clone()
        elif self.audio2_mode == "glottal_lpc":
            x2 = self.forward_glottal(x)
        x1 = self.encoder1(x.unsqueeze(1))
        x2 = self.encoder2(x2.unsqueeze(1))

        x1, _, _ = self.quantizer1(x1)
        x2, _, _ = self.quantizer2(x2)

        # print(f"x1: {x1.shape}, x2: {x2.shape}, concatenated: {torch.cat([x1, x2], dim=1).shape}")

        return torch.cat([x1, x2], dim=1).transpose(1, 2)