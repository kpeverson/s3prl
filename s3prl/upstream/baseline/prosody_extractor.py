
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
# from torchaudio.functional import compute_deltas
from torchaudio.transforms import MFCC, MelScale, Resample, Spectrogram
import torchcrepe

############
# CONSTANT #
############
N_SAMPLED_PSEUDO_WAV = 2

def get_prosody_extractor(config):
    # feat_list = config["feat_list"]
    prosody_extractor = ProsodyExtractor(config)
    return prosody_extractor

class ProsodyExtractor(torch.nn.Module):
    def __init__(self, config):
        super(ProsodyExtractor, self).__init__()
        self.config = config
        self.feat_list = config["prosody"]["feat_list"]
        self.sample_rate = config.get("sr", 16000)
        self.feat_rate = config["prosody"]["feat_rate"]
        self.dummy = config["prosody"].get("dummy", False)
        
        # if "pitch" in self.feat_list or "periodicity" in self.feat_list:
        if not self.dummy:
            if "c1" in self.feat_list:
                self.mfcc_trans = MFCC(
                    sample_rate=self.sample_rate,
                    n_mfcc=config["prosody"]["c1"]["num_ceps"],
                    log_mels=config["prosody"]["c1"]["log_mels"],
                    melkwargs={
                        "n_mels": config["prosody"]["c1"]["num_mels"],
                        "hop_length": int(self.sample_rate / self.feat_rate),
                    }
                ).to("cpu")
                self.melspec_trans = torchaudio.transforms.MelSpectrogram(
                        # n_fft=config["prosody"]["c1"]["n_fft"],
                        # win_length=config["prosody"]["c1"]["win_length"],
                        sample_rate=self.sample_rate,
                        n_mels=config["prosody"]["c1"]["num_mels"],
                        hop_length=int(self.sample_rate / self.feat_rate),
                        # power=None,  # return complex spectrogram
                    )
                print(f"mfcc transform initialized with sample_rate={self.sample_rate} ({type(self.sample_rate)}), feat_rate={self.feat_rate} ({type(self.feat_rate)}), num_ceps={config['prosody']['c1']['num_ceps']} ({type(config['prosody']['c1']['num_ceps'])}), log_mels={config['prosody']['c1']['log_mels']} ({type(config['prosody']['c1']['log_mels'])}), num_mels={config['prosody']['c1']['num_mels']} ({type(config['prosody']['c1']['num_mels'])})")

        self.register_buffer(
            "_pseudo_wavs", torch.randn(N_SAMPLED_PSEUDO_WAV, self.sample_rate)
        )

    def normalize_by_periodicity(self, f0, periodicity):
        """
        Mean-normalize f0, with mean f0 computed as weighted average by periodicity.
        """
        f0_mean = (f0 * periodicity).sum() / periodicity.sum()
        f0 = f0 - f0_mean
        return f0

    def extract_f0_periodicity(self, wav):
        pitch_q = self.config["prosody"]["pitch"]["pitch_q"]
        fmin = self.config["prosody"]["pitch"]["fmin"]
        fmax = self.config["prosody"]["pitch"]["fmax"]
        def _reshape(arr, q):
            b = arr.shape[0]
            l = arr.shape[1]
            arr = arr[:, :int(l//q)*q]
            arr = arr.reshape(b, l//q, q)
            arr = arr.mean(dim=-1)
            return arr
        
        pitch_hop_length = int(self.sample_rate / (self.feat_rate * pitch_q))
        f0, periodicity = torchcrepe.predict(
            wav.unsqueeze(0),
            self.sample_rate,
            pitch_hop_length,
            fmin,
            fmax,
            "full",
            batch_size=2048,
            device=wav.device,
            return_periodicity=True,
        )
        f0 = _reshape(f0, pitch_q) if pitch_q > 1 else f0
        periodicity = _reshape(periodicity, pitch_q) if pitch_q > 1 else periodicity
        if self.config["prosody"]["pitch"]["log"]:
            f0 = torch.log(f0 + 1e-6)
        if self.config["prosody"]["pitch"]["utterance_normalization"]:
            f0 = self.normalize_by_periodicity(f0, periodicity)
        return f0, periodicity

    def extract_c1(self, wav):
        """
        Args:
            wav (torch tensor, shape (T_w,)): waveform

        Returns:
            c1 (torch tensor, shape (T_f,)): c1 (first MFCC coefficient)
        """
        mfcc = self.mfcc_trans.cpu()(wav)
        c1 = mfcc[1, :]
        return c1
    
    def compute_delta(self, x):
        """
        Args:
            x (torch tensor, shape (T_f,)): feature

        Returns:
            delta_x (torch tensor, shape (T_f,)): delta features
        """
        delta_x = torch.cat(
            (
                x[0].unsqueeze(0),
                x[1:] - x[:-1],
            ), 0
        )
        return delta_x
    
    def forward(self, wav):
        """
        Args:
            wav (torch tensor, shape (T_w,)): waveform

        Returns:
            features (list of torch tensors, shape (B, T_f, C)): list of extracted features for each waveform, dimension C
        """
        if self.dummy:
            # return zeros
            feats = torch.zeros(
                (int(wav.shape[0]*self.feat_rate/self.sample_rate), len(self.feat_list)), device=wav.device
            )
        else:
            f0, periodicity = self.extract_f0_periodicity(wav)
            delta_f0 = self.compute_delta(f0)
            c1 = self.extract_c1(wav.cpu()).to(device=wav.device).unsqueeze(0)
            min_len = min(
                f0.shape[1], periodicity.shape[1], c1.shape[1], delta_f0.shape[1]
            )
            feats = torch.cat(
                [
                    f0[:, :min_len],
                    delta_f0[:, :min_len],
                    c1[:, :min_len],
                    periodicity[:, :min_len],
                ], dim=0
            ).transpose(0, 1)
        return feats
