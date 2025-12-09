import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np 
import librosa
from librosa.util import find_files
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
import json
# import pyworld as pw
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

from ...dataset.glottal_extraction import GlottalExtractor

# from .reaper import REAPERExtractor


CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')
SAMPLE_RATE = 16000

DEBUG = False
USEYAAPT = True
USEREAPER = False
USEBIN = False
# LibriTTS pitch reconstruction
class PitchDataset(Dataset):
    def __init__(self, mode, h5_dir, meta_data, glottal_kwargs, upstream_rate=160, log_pitch=True, **kwargs):
        
        self.mode = mode
        self.h5_dir = h5_dir
        if mode == 'train':
            self.h5_path = os.path.join(h5_dir, f"{mode}-clean-100.h5")
        else:
            self.h5_path = os.path.join(h5_dir, f"{mode}-clean.h5")
        self.h5_file = None
        self.meta_data = meta_data
        self.upstream_rate = upstream_rate
        self.fp = 1000 // (SAMPLE_RATE // self.upstream_rate)
        self.log_pitch = log_pitch
        self.return_glottal = glottal_kwargs.get('return_glottal', False)
        if self.return_glottal:
            self.glottal_extractor = GlottalExtractor(
                sr=SAMPLE_RATE,
                lpc_window_size=glottal_kwargs.get('lpc_window_size', 0.025),
                lpc_window_stride=glottal_kwargs.get('lpc_window_stride', 0.010),
                lpc_order=glottal_kwargs.get('lpc_order', 16),
                lpc_window=glottal_kwargs.get('lpc_window', 'hamming'),
                lpf_cutoff=glottal_kwargs.get('lpf_cutoff', 1000),
                lpf_order=glottal_kwargs.get('lpf_order', 4),
                half_band_signal=glottal_kwargs.get('half_band_signal', False)
            )

        if DEBUG:
            print("Frame Period: ", self.fp)
        self.usage_list = []
        with open(f"{self.meta_data}/{mode}-filtered.txt", "r", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    continue
                # remove ".wav" suffix from each line
                line = line.split(".wav")[0]
                self.usage_list.append(line.strip())

        cache_path = os.path.join(CACHE_PATH, f'{mode}-{self.fp}.pkl')
        if os.path.isfile(cache_path):
            print(f'[Pitch Dataset] - Loading file paths from {cache_path}')
            with open(cache_path, 'rb') as cache:
                dataset = pickle.load(cache)
        else:
            dataset  = getattr(self, f"build_{mode}_dataset")()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as cache:
                pickle.dump(dataset, cache)
        print(f'[Pitch Dataset] - there are {len(dataset)} files found')

        self.dataset = dataset

        cache_path = os.path.join(CACHE_PATH, f'{mode}-labels-{self.fp}.pkl')
        if USEREAPER:
            self.errors = 0
            cache_path = os.path.join(CACHE_PATH, f'{mode}-reaper-labels-{self.fp}.pkl')
            default_cache_path = os.path.join(CACHE_PATH, f'{mode}-yaapt-labels-{self.fp}.pkl')
            with open(default_cache_path, 'rb') as f:
                self.default_labels = pickle.load(f)
        if USEYAAPT:
            cache_path = os.path.join(CACHE_PATH, f'{mode}-yaapt-labels-{self.fp}.pkl')
        if USEBIN:
            cache_path = cache_path.replace('labels', 'bin_labels')
        if os.path.isfile(cache_path):
            print(f'[Pitch Dataset] - Loading labels from {cache_path}')
            with open(cache_path, 'rb') as cache:
                all_labels = pickle.load(cache)
        else:
            all_labels = self.build_label(self.dataset)
            with open(cache_path, 'wb') as cache:
                pickle.dump(all_labels, cache)

        self.label = all_labels

        cache_path = os.path.join(CACHE_PATH, f'{mode}-stats-{self.fp}.pkl')
        if USEREAPER:
            cache_path = os.path.join(CACHE_PATH, f'{mode}-reaper-stats-{self.fp}.pkl')
        if USEYAAPT:
            cache_path = os.path.join(CACHE_PATH, f'{mode}-yaapt-stats-{self.fp}.pkl')
        if os.path.isfile(cache_path):
            print(f'[Pitch Dataset] - Loading stats from {cache_path}')
            with open(cache_path, 'rb') as cache:
                all_stats = pickle.load(cache)
        else:
            all_stats = self.build_stats()
            with open(cache_path, 'wb') as cache:
                pickle.dump(all_stats, cache)

        self.norm_stat = all_stats

    def _get_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file

    # def name2path(self, name):
    #     prefix = name.split('_')
    #     if self.mode == "train":
    #         return f"{self.root}/train-clean-100/{prefix[0]}/{prefix[1]}/{name}"
    #     elif self.mode == "dev":
    #         return f"{self.root}/dev-clean/{prefix[0]}/{prefix[1]}/{name}"
    #     elif self.mode == "test":
    #         return f"{self.root}/test-clean/{prefix[0]}/{prefix[1]}/{name}"
    #     else:
    #         raise NotImplementedError

    def build_label(self, path_list):
        h5_file = self._get_h5_file()
        y = {}
        if USEREAPER:
            os.makedirs("./downstream/pitch_libritts/.tmp1", exist_ok=True)
            extractor = REAPERExtractor()
        for path in tqdm.tqdm(path_list, desc="Pitch extraction"):
            # wav_path = self.name2path(path)

            if USEYAAPT:
                # pYAAPT
                # signal = basic.SignalObj(wav_path)
                wav = h5_file[path][:]
                signal = basic.SignalObj(wav, SAMPLE_RATE)
                dio_len = int(signal.size * 1000 / signal.fs / self.fp) + 1
                f0_pad = np.zeros(dio_len, dtype=np.float64)
                try:
                    pitch = pYAAPT.yaapt(signal, **{'f0_min' : 71.0, 'f0_max' : 800.0, 'frame_space' : self.fp})
                    f0 = pitch.samp_values.astype(np.float64)
                    f0_pad[:len(f0)] = f0
                    y[path] = f0_pad
                except:
                    print("Error detected: ", wav_path)
                    y[path] = f0_pad
            elif USEREAPER:
                # REAPER
                extractor = REAPERExtractor()
                # signal = basic.SignalObj(wav_path)
                wav = h5_file[path][:]
                signal = basic.SignalObj(wav, SAMPLE_RATE)
                dio_len = int(signal.size * 1000 / SAMPLE_RATE / self.fp) + 1
                f0_pad = np.zeros(dio_len, dtype=np.float64)
                try:
                    extractor.exec(
                        wav_path=wav_path,
                        output_path=f"./downstream/pitch_libritts/.tmp1/{path[:-4]}",
                        fp=self.fp / 1000
                    )
                    res = extractor.parse_f0_file(f"./downstream/pitch_libritts/.tmp1/{path[:-4]}.f0")
                    f0_pad[:len(res)] = np.array(res, dtype=np.float64)
                    y[path] = f0_pad
                except:
                    self.errors += 1
                    print("Error detected: ", wav_path)
                    y[path] = self.default_labels[path]
            else:
                # pyWorld
                wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
                pitch, t = pw.dio(wav.astype(np.float64), SAMPLE_RATE, frame_period=self.fp)
                pitch = pw.stonemask(wav.astype(np.float64), pitch, t, SAMPLE_RATE)
                y[path] = pitch
        if USEREAPER:
            print("ERROR: ", self.errors)
        return y
    
    def build_stats(self):
        pitches = []
        for path, pitch in tqdm.tqdm(self.label.items()):
            for p in pitch:
                if p == 0:
                    continue
                if self.log_pitch:
                    p = np.log(p)
                pitches.append(p)
        n = len(pitches)
        mean = sum(pitches) / n
        variance = sum([((x - mean) ** 2) for x in pitches]) / n
        std = variance ** 0.5
        return (mean, std)
    
    def build_train_dataset(self):
        dataset = self.usage_list
        if DEBUG:
            dataset = dataset[:100]
        print("finish searching training set wav")
        return dataset

    def build_dev_dataset(self):
        dataset = self.usage_list
        if DEBUG:
            dataset = dataset[:100]
        print("finish searching dev set wav")
        return dataset

    def build_test_dataset(self):
        dataset = self.usage_list
        if DEBUG:
            dataset = dataset[:100]
        print("finish searching test set wav")
        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        path = self.dataset[idx]
        h5_file = self._get_h5_file()
        if h5_file is not None:
            wav = h5_file[path][:]
            if self.return_glottal:
                wav = self.glottal_extractor.extract(torch.from_numpy(wav), idx)
        else:
            raise NotImplementedError("H5 file is required for loading wav data since LibriTTS-R audios have been converted to .h5 format.")
            # wav, _ = librosa.load(self.name2path(path), sr=SAMPLE_RATE)
        
        
        return np.float32(wav), torch.FloatTensor(self.label[path]).view(-1, 1)
        
    def collate_fn(self, samples):
        return zip(*samples)
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()