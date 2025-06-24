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

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')

# BU Radio Corpus break indices classification dataset
class BreakIdxDataset(Dataset):
    def __init__(self, mode, corpus_dir, meta_data, max_timestep=None, return_glottal=False, sr=16000, **kwargs):
        self.root = corpus_dir
        self.meta_data = meta_data
        self.split_list = open(self.meta_data, "r").readlines()
        self.max_timestep = max_timestep
        self.sr = sr

        cache_path = os.path.join(CACHE_PATH, f'{mode}.pkl')
        if os.path.isfile(cache_path):
            print(f'[BreakIdxDataset] - Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                self.dataset = pickle.load(cache)
        else:
            self.dataset = eval("self.{}".format(mode))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump(self.dataset, cache)
        print(f'[BreakIdxDataset] - there are {len(self.dataset)} files found')
        # self.dataset = dataset
        self.times_labels = self.get_times_labels(self.dataset)

        self.return_glottal = return_glottal
        if return_glottal:
            self.lpc_order = kwargs.get("lpc_order", 16)
            self.lpc_window = kwargs.get("lpc_window", "hamming")
            self.lpc_window_size = kwargs.get("lpc_window_size", 0.025)
            self.lpc_window_stride = kwargs.get("lpc_window_stride", 0.01)
            self.energy_threshold = kwargs.get("energy_threshold", 1e-4)
            self.glottal_lpf_cutoff = kwargs.get("glottal_lpf_cutoff", 1000)

    def train(self):
        dataset = []
        print(f"[BreakIdxDataset] - Loading training data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[0]
            if int(index) == 1:
                x = pair[1]
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and os.path.exists(os.path.join(self.root, f"{x}.brk")):
                    dataset.append(x)
        print(f"[BreakIdxDataset] - {len(dataset)} training files found")
        return dataset
    
    def dev(self):
        dataset = []
        print(f"[BreakIdxDataset] - Loading development data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[0]
            if int(index) == 2:
                x = pair[1]
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and os.path.exists(os.path.join(self.root, f"{x}.brk")):
                    dataset.append(x)
        print(f"[BreakIdxDataset] - {len(dataset)} development files found")
        return dataset
    
    def test(self):
        dataset = []
        print(f"[BreakIdxDataset] - Loading test data from {self.root}")
        for line in tqdm.tqdm(self.split_list):
            pair = line.strip().split()
            index = pair[0]
            if int(index) == 3:
                x = pair[1]
                if os.path.exists(os.path.join(self.root, f"{x}.sph")) and os.path.exists(os.path.join(self.root, f"{x}.brk")):
                    dataset.append(x)
        print(f"[BreakIdxDataset] - {len(dataset)} test files found")
        return dataset
    
    def get_times_labels(self, dataset):
        all_times_labels = []
        for path in dataset:
            times_labels = []
            brk_path = os.path.join(self.root, f"{path}.brk")
            with open(brk_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
            # filter only lines with 3 columns
            lines = [line for line in lines if len(line.split()) == 3]
            # filter only lines with format ""
            for line in lines:
                boundary_time, _, break_idx = line.split()
                boundary_time = float(boundary_time)
                break_idx = int(break_idx[0])
                # assert break_idx in [1, 2, 3, 4]
                if break_idx in [0, 1, 2, 3, 4]:
                    times_labels.append((boundary_time, break_idx))
            all_times_labels.append(times_labels)
        num_breaks = [len(times_labels) for times_labels in all_times_labels]
        print(f"[BreakIdxDataset] - {len(all_times_labels)} files with times and labels found")
        print(f"[BreakIdxDataset] - number of breaks per file, min: {min(num_breaks)}, max: {max(num_breaks)}, avg: {np.mean(num_breaks)}")
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
        wav = self.load_audio(idx)
        length = wav.shape[0]
        
        if self.max_timestep is not None:
            if length > self.max_timestep:
                start = random.randint(0, int(length - self.max_timestep))
                wav = wav[start:start + self.max_timestep]
                length = self.max_timestep
        
        times_labels = self.times_labels[idx]
        times = [t[0] for t in times_labels]
        labels = [t[1] for t in times_labels]
        labels = self.map_labels(labels)
        return wav, times, labels, os.path.basename(self.dataset[idx])
    
    def collate_fn(self, samples):
        return zip(*samples)