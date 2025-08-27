import atexit
import signal as sig
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
from librosa.util import find_files
import scipy
from scipy import signal
from torchaudio import load
from torch import nn
import os
import re
import random
import pickle
import torchaudio
import sys
import time
import glob
import tqdm
from pathlib import Path

# CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')
EXCLUDE_IDS = ["data/f3a/labnews/j/radio/f3ajrlp2", "data/f2b/radio/s18/f2bs18p8"]

# BU Radio Corpus break indices classification dataset
class TonesDataset(Dataset):
    def __init__(self, mode, corpus_dir, meta_data, max_timestep=None, return_glottal=False, sr=16000, h5_path=None, **kwargs):
        self.root = corpus_dir
        self.meta_data = meta_data
        self.split_list = open(self.meta_data, "r").readlines()
        self.max_timestep = max_timestep
        self.sr = sr

        self.h5_path = h5_path
        self.h5_file = None
        if self.h5_path is not None:
            self._register_cleanup()

        # cache_path = os.path.join(CACHE_PATH, f'{mode}.pkl')
        # if os.path.isfile(cache_path):
        #     print(f'[TonesDataset] - Loading file paths from {cache_path}')
        #     with open(cache_path, 'rb') as cache:
        #         self.dataset = pickle.load(cache)
        # else:
        self.dataset = eval("self.{}".format(mode))()
            # os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            # with open(cache_path, 'wb') as cache:
            #     pickle.dump(self.dataset, cache)
        # print(f'[TonesDataset] - there are {len(self.dataset)} files found')
        # self.dataset = dataset
        self.times_labels = self.get_times_labels(self.dataset)
        all_labels = {i: 0 for i in range(2)}  # Initialize all labels to 0
        for tls in self.times_labels:
            for _, _, l in tls:
                all_labels[l] += 1
        print(f"[TonesDataset] - labels distribution: {all_labels}")

        self.return_glottal = return_glottal
        if return_glottal:
            self.lpc_order = kwargs.get("lpc_order", 16)
            self.lpc_window = kwargs.get("lpc_window", "hamming")
            self.lpc_window_size = kwargs.get("lpc_window_size", 0.025)
            self.lpc_window_stride = kwargs.get("lpc_window_stride", 0.01)
            self.energy_threshold = kwargs.get("energy_threshold", 1e-4)
            self.glottal_lpf_cutoff = kwargs.get("glottal_lpf_cutoff", 1000)

    def _ensure_open(self):
        if self.h5_path is not None and self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            print(f"[TonesDataset] - Opened h5 file {self.h5_path}")

    def _cleanup(self):
        if self.h5_file is not None:
            print(f"[TonesDataset] - Closing h5 file {self.h5_path}")
            self.h5_file.close()
            self.h5_file = None

    def _register_cleanup(self):
        atexit.register(self._cleanup)

        def signal_handler(sig, frame):
            self._cleanup()
            raise KeyboardInterrupt

        sig.signal(sig.SIGINT, signal_handler)
        sig.signal(sig.SIGTERM, signal_handler)

    def train(self):
        dataset = []
        print(f"[TonesDataset] - Loading training data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[1].strip()
            if int(index) == 1:
                x = pair[0].strip()
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ton")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ala")):
                    if x in EXCLUDE_IDS:
                        # print(f"[TonesDataset] - Excluding {x} from training set")
                        continue
                    dataset.append(x)
        print(f"[TonesDataset] - {len(dataset)} training files found")
        return dataset
    
    def dev(self):
        dataset = []
        print(f"[TonesDataset] - Loading development data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[1].strip()
            if int(index) == 2:
                x = pair[0].strip()
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ton")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ala")):
                    if x in EXCLUDE_IDS:
                        # print(f"[TonesDataset] - Excluding {x} from dev set")
                        continue
                    dataset.append(x)
        print(f"[TonesDataset] - {len(dataset)} development files found")
        return dataset
    
    def test(self):
        dataset = []
        print(f"[TonesDataset] - Loading test data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[1].strip()
            if int(index) == 3:
                x = pair[0].strip()
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ton")) and \
                    os.path.exists(os.path.join(self.root, f"{x}.ala")):
                    if x in EXCLUDE_IDS:
                        # print(f"[TonesDataset] - Excluding {x} from test set")
                        continue
                    dataset.append(x)
        print(f"[TonesDataset] - {len(dataset)} test files found")
        return dataset
    
    def get_times_labels(self, dataset):
        all_times_labels = []
        ala_frame_size = 0.01 # 10ms frame size for .ala files
        for path in dataset:
            times_tones = []
            ton_path = os.path.join(self.root, f"{path}.ton")
            ala_path = os.path.join(self.root, f"{path}.ala")
            with open(ton_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
            # filter only lines with 3 columns
            lines = [line for line in lines if len(line.split()) == 3]
            # filter only lines with format ""
            for line in lines:
                time, _, ton = line.split()
                time = float(time)
                times_tones.append((time, ton))
            print(f"[TonesDataset] - {path}: times_tones: {times_tones}")
            # load words
            times_labels = []
            with open(ala_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
            # filter only lines with 3 columns
            lines = [line.split() for line in lines if len(line.split()) == 3]
            lines = [(line[0], int(line[1]), int(line[1])+int(line[2])) for line in lines] # (phoneme, start_idx, end_idx)
            vowel_idxs = [i for i, line in enumerate(lines) if any([vowel in line[0] for vowel in 'AEIOU'])]
            for i, vidx in enumerate(vowel_idxs):
                # st idx
                if vidx == 0:
                    st_idx = 0.0
                elif i!=0 and vowel_idxs[i-1] == vidx-1:
                    # two adjacent phonemes
                    st_idx = lines[vidx][1]
                else:
                    # standard case - center of previous phoneme
                    st_idx = (lines[vidx-1][1] + lines[vidx-1][2]) / 2.0

                # et idx
                if vidx == len(lines) - 1:
                    et_idx = lines[-1][2]
                elif i!=len(vowel_idxs)-1 and vowel_idxs[i+1] == vidx+1:
                    # two adjacent phonemes
                    et_idx = lines[vidx][2]
                else:
                    # standard case - center of following phoneme
                    et_idx = (lines[vidx+1][1] + lines[vidx+1][2]) / 2.0
                st = st_idx * ala_frame_size
                et = et_idx * ala_frame_size
                tones_in_phone = [tt[1] for tt in times_tones if st <= tt[0] <= et]
                if any(["*" in ton for ton in tones_in_phone]):
                    times_labels.append((st, et, 1))
                else:
                    times_labels.append((st, et, 0))
            #     print(f"[TonesDataset] - {path}: {lines[vidx][0]} from idx {st_idx} {st:.2f}s to idx {et_idx} {et:.2f}s, tones: {tones_in_phone}, label: {times_labels[-1][2]}")
            # exit()
            all_times_labels.append(times_labels)
        num_breaks = [len(times_labels) for times_labels in all_times_labels]
        print(f"[TonesDataset] - {len(all_times_labels)} files, number of breaks per file, min: {min(num_breaks)}, max: {max(num_breaks)}, avg: {np.mean(num_breaks)}")
        return all_times_labels

    def __len__(self):
        return len(self.dataset)
    
    def lpf(self, x, sr):
        sos = signal.butter(4, self.glottal_lpf_cutoff, "low", fs=sr, output="sos")
        x = signal.sosfiltfilt(sos, x)
        return x
    
    def inverse_filter(self, x, a, energy_threshold=1e-4):
        # if np.sum(x**2) < energy_threshold:
        #     print("Energy is too low, skipping inverse filtering")
        #     return x
        x_hat = signal.lfilter(
            np.hstack([[0], -1 * a[1:]]), [1], x
        )
        glottal = x - x_hat
        return glottal
    
    def forward_glottal(self, x, sr, idx):
        lpc_window_size = int(sr * self.lpc_window_size)
        lpc_window_stride = int(sr * self.lpc_window_stride)
        
        x = x.numpy()
        glottal_source = np.zeros_like(x)
        frames = librosa.util.frame(x, frame_length=lpc_window_size, hop_length=lpc_window_stride).T
        if self.lpc_window == "hamming":
            window = np.hamming(lpc_window_size)
        else:
            raise ValueError(f"Unsupported window type: {self.lpc_window}")
        
        for i, frame in enumerate(frames):
            frame = frame*window
            a = librosa.lpc(frame, order=self.lpc_order)
            frame_glottal_source = self.inverse_filter(frame, a)
            glottal_source[i*lpc_window_stride:i*lpc_window_stride+lpc_window_size] += frame_glottal_source
        
        if np.isnan(glottal_source).any():
            print(f"Bad glottal source for {idx}: {np.isnan(glottal_source).sum()} NaNs")
        elif np.abs(glottal_source).max() > 50.0:
            print(f"Bad glottal source for {idx}: {np.abs(glottal_source).max()}")

        if self.glottal_lpf_cutoff is not None:
            glottal_source = self.lpf(glottal_source, sr)

        # print(f"x shape: {x.shape}, glottal source shape: {glottal_source.shape}")
        return glottal_source

    def load_audio(self, idx):
        sph_path = os.path.join(self.root, f"{self.dataset[idx]}.sph")
        wav, sr = torchaudio.load(sph_path)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(wav)
            sr = self.sr
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=False)
        wav = wav.squeeze(0)
        if self.return_glottal:
            wav = self.forward_glottal(wav, sr, idx)
        else:
            wav = wav.numpy()

        return wav

    def __getitem__(self, idx):
        utt_id = os.path.basename(self.dataset[idx])

        wav = self.load_audio(idx)
        length = wav.shape[0]
        
        if self.max_timestep is not None:
            if length > self.max_timestep:
                start = random.randint(0, int(length - self.max_timestep))
                wav = wav[start:start + self.max_timestep]
                length = self.max_timestep
        
        times_labels = self.times_labels[idx]
        sts = [t[0] for t in times_labels]
        ets = [t[1] for t in times_labels]
        labels = [t[2] for t in times_labels]
        # labels = self.map_labels(labels)
        if len(sts)==0:
            print(f"[TonesDataset] - No breaks found for {self.dataset[idx]}")
        
        self._ensure_open()
        if self.h5_file is not None:
            # load feats from h5 file
            h5_feats = torch.from_numpy(self.h5_file[utt_id+".sph"][:])
        else:
            # h5_feats = torch.zeros_like(wav)
            h5_feats = torch.zeros((len(wav), 1)) # placeholder if no h5 file is provided
        
        return wav, sts, ets, labels, h5_feats, utt_id
    
    def collate_fn(self, samples):
        return zip(*samples)