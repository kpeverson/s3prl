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


# Voxceleb 1 Speaker Identification
class SpeakerClassifiDataset(Dataset):
    def __init__(self, mode, file_path, meta_data, max_timestep=None, return_glottal=False, **kwargs):

        self.root = file_path
        self.speaker_num = 1251
        self.meta_data =meta_data
        self.max_timestep = max_timestep
        self.usage_list = open(self.meta_data, "r").readlines()

        cache_path = os.path.join(CACHE_PATH, f'{mode}.pkl')
        if os.path.isfile(cache_path):
            print(f'[SpeakerClassifiDataset] - Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                dataset = pickle.load(cache)
        else:
            dataset = eval("self.{}".format(mode))()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump(dataset, cache)
        print(f'[SpeakerClassifiDataset] - there are {len(dataset)} files found')

        self.dataset = dataset
        self.label = self.build_label(self.dataset)

        self.return_glottal = return_glottal
        if return_glottal:
            self.lpc_order = kwargs.get("lpc_order", 16)
            self.lpc_window = kwargs.get("lpc_window", "hamming")
            self.lpc_window_size = kwargs.get("lpc_window_size", 0.025)
            self.lpc_window_stride = kwargs.get("lpc_window_stride", 0.01)
            self.energy_threshold = kwargs.get("energy_threshold", 1e-4)
            self.glottal_lpf_cutoff = kwargs.get("glottal_lpf_cutoff", 1000)

    # file_path/id0001/asfsafs/xxx.wav
    def build_label(self, train_path_list):

        y = []
        for path in train_path_list:
            id_string = path.split("/")[-3]
            y.append(int(id_string[2:]) - 10001)

        return y
    
    @classmethod
    def label2speaker(self, labels):
        return [f"id{label + 10001}" for label in labels]
    
    def train(self):

        dataset = []
        print("search specified wav name for training set")
        for string in tqdm.tqdm(self.usage_list):
            pair = string.split()
            index = pair[0]
            if int(index) == 1:
                x = list(self.root.glob("wav/" + pair[1]))
                dataset.append(str(x[0]))
        print("finish searching training set wav")
                
        return dataset
        
    def dev(self):

        dataset = []
        print("search specified wav name for dev set")
        for string in tqdm.tqdm(self.usage_list):
            pair = string.split()
            index = pair[0]
            if int(index) == 2:
                x = list(self.root.glob("wav/" + pair[1]))
                dataset.append(str(x[0])) 
        print("finish searching dev set wav")

        return dataset       

    def test(self):

        dataset = []
        print("search specified wav name for test set")
        for string in tqdm.tqdm(self.usage_list):
            pair = string.split()
            index = pair[0]
            if int(index) == 3:
                x = list(self.root.glob("wav/" + pair[1]))
                dataset.append(str(x[0])) 
        print("finish searching test set wav")

        return dataset

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

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.dataset[idx])
        wav = wav.squeeze(0)
        length = wav.shape[0]
        if self.return_glottal:
            glottal_wav = self.forward_glottal(wav, sr, idx)

        if self.max_timestep !=None:
            if length > self.max_timestep:
                start = random.randint(0, int(length-self.max_timestep))
                wav = wav[start:start+self.max_timestep]
                if self.return_glottal:
                    glottal_wav = glottal_wav[start:start+self.max_timestep]
                length = self.max_timestep

        def path2name(path):
            return Path("-".join((Path(path).parts)[-3:])).stem

        path = self.dataset[idx]
        if self.return_glottal:
            return glottal_wav, self.label[idx], path2name(path)
        return wav.numpy(), self.label[idx], path2name(path)
        
    def collate_fn(self, samples):
        return zip(*samples)
