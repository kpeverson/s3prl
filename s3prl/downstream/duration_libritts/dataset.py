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
from transformers import AutoTokenizer, AutoModel
import json
import textgrid

from ...dataset.glottal_extraction import GlottalExtractor

CACHE_PATH = os.path.join(os.path.dirname(__file__), '.cache/')
SAMPLE_RATE = 16000
LIBRITTSR_SR = 24000

class DurationDataset(Dataset):
    def __init__(self, mode, corpus_dir, meta_data_dir, glottal_kwargs, upstream_rate=160, use_spkr_embeds=False, **kwargs):

        self.mode = mode
        self.corpus_dir = corpus_dir
        if mode == 'train':
            self.h5_path = os.path.join(corpus_dir, f'{mode}-clean-100.h5')
        else:
            self.h5_path = os.path.join(corpus_dir, f'{mode}-clean.h5')
        self.h5_file = None
        self.meta_data_dir = meta_data_dir
        self.upstream_rate = upstream_rate
        self.min_frame_size = 1.0 / (SAMPLE_RATE / self.upstream_rate)
        print(f"[DurationDataset] - min_frame_size: {self.min_frame_size} seconds")

        self.use_spkr_embeds = use_spkr_embeds
        if self.use_spkr_embeds:
            self.spkr_embeds_h5_path = os.path.join(corpus_dir, "titanet_large_spkr_embs.h5")
        self.spkr_embeds_h5_file = None

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

        usage_list = []
        with open(f"{self.meta_data_dir}/{mode}-filtered.txt", "r", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    continue
                # remove ".wav" suffix from each line
                line = line.split(".wav")[0]
                usage_list.append(line.strip())
        self.dataset = usage_list

    def _get_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file
    
    def __len__(self):
        return len(self.dataset)

    def filter_short_durations(self, times_durs):
        filtered = [
            [td for td in tds if td[2] >= self.min_frame_size]
            for tds in times_durs
        ]
        return filtered

class PhoneDurationDataset(DurationDataset):
    def __init__(self, mode, corpus_dir, meta_data_dir, vocab_path, glottal_kwargs, intra_word_pos=False, upstream_rate=160, use_spkr_embeds=False, **kwargs):
        super().__init__(mode, corpus_dir, meta_data_dir, glottal_kwargs, upstream_rate=upstream_rate, use_spkr_embeds=use_spkr_embeds, **kwargs)
        self.intra_word_pos = intra_word_pos
        cache_path = os.path.join(CACHE_PATH, f"phone/{self.mode}_times_durs.pkl")
        if os.path.isfile(cache_path):
            print(f"[PhoneDurationDataset] - loading cached {self.mode} times_durs from {cache_path}")
            with open(cache_path, "rb") as f:
                times_durs = pickle.load(f)
        else:
            times_durs = self.get_times_durs(self.dataset)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            print(f"[PhoneDurationDataset] - saving cached {self.mode} times_durs to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(times_durs, f)
        # filter out examples with duration less than min_frame_size
        self.times_durs = self.filter_short_durations(times_durs)

        self.vocab_path = vocab_path
        self.get_vocab()
        self.phone_stress = ['NONE', '0', '1', '2']

    def get_vocab(self):
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.arpabet_phones = list(set([line.strip().replace("0", "").replace("1", "").replace("2", "") for line in f if line.strip()!=""])) + ["sil"]
        if self.mode == "train":
            print(f"[DurationDataset] - self.arpabet_phones: {self.arpabet_phones}")

    def parse_phones(self, tg):
        phone_tier = tg.getFirst("phones")
        # times_phones = [
        #     (interval.minTime, interval.maxTime, interval.maxTime-interval.minTime, interval.mark.strip()) for interval in phone_tier
        #     if interval.mark.strip() != ""
        # ]
        if self.intra_word_pos:
            # add intra-word position to phones
            word_tier = tg.getFirst("words")
            words = [
                (interval.minTime, interval.maxTime, interval.mark.strip()) for interval in word_tier
                if interval.mark.strip() != ""
            ]
            phones = [
                (interval.minTime, interval.maxTime, interval.mark.strip()) for interval in phone_tier
            ]

            phones_by_word = []
            phone_idx = 0
            n_phones = len(phone_tier)
            for w_start, w_end, _ in words:
                # get phones within this word
                this_word_phones = []

                # move phone_idx to the first phone within this word
                while phone_idx < n_phones and phones[phone_idx][1] <= w_start:
                    phone_idx += 1

                # collect phones within this word
                while phone_idx < n_phones and phones[phone_idx][0] < w_end:
                    this_word_phones.append(phones[phone_idx])
                    phone_idx += 1

                phones_by_word.append(this_word_phones)

            times_phones = [
                (start, end, end-start, phone, pos, len(word_phones))
                for word_phones in phones_by_word
                for pos, (start, end, phone) in enumerate(word_phones)
            ]
        else:
            times_phones = [
                (interval.minTime, interval.maxTime, interval.maxTime-interval.minTime, interval.mark.strip()) for interval in phone_tier
                if interval.mark.strip() != ""
            ]

        return times_phones
    
    def get_times_durs(self, dataset):
        if self.mode == "train":
            align_dir = os.path.join(self.corpus_dir, f"alignments/{self.mode}_clean_100")
        else:
            align_dir = os.path.join(self.corpus_dir, f"alignments/{self.mode}_clean")
        all_times_durs = []
        for item in tqdm.tqdm(dataset, desc=f"loading {self.mode} alignments", total=len(dataset)):
            align_path = os.path.join(
                align_dir,
                item+".TextGrid"
            )
            tg = textgrid.TextGrid.fromFile(align_path)
            all_times_durs.append(self.parse_phones(tg))
        num_durs = [len(tds) for tds in all_times_durs]
        print(f"[PhoneDurationDataset] - {len(all_times_durs)} files, number of phones: {min(num_durs)} ~ {max(num_durs)}, average: {sum(num_durs)/len(num_durs):.2f}")
        return all_times_durs

    def collate_fn(self, samples):
        return zip(*samples)
    
    def __getitem__(self, idx):
        path = self.dataset[idx]
        h5_file = self._get_h5_file()
        if h5_file is not None:
            wav = h5_file[path][:]
            # resample from 24000 to 16000
            wav = librosa.resample(wav.astype(np.float32), orig_sr=LIBRITTSR_SR, target_sr=SAMPLE_RATE)
            if self.return_glottal:
                wav = self.glottal_extractor.extract(torch.from_numpy(wav), idx)
        else:
            raise NotImplementedError("H5 file is required for loading wav data since LibriTTS-R audios have been converted to .h5 format.")
        
        times_durs = self.times_durs[idx]
        sts = [t[0] for t in times_durs]
        ets = [t[1] for t in times_durs]
        durs = [t[2] for t in times_durs]
        
        phones = [t[3] if t[3] != '' else 'sil' for t in times_durs]
        phones_wout_stress = [p.replace('0', '').replace('1', '').replace('2', '') for p in phones]
        phones_idxs = [self.arpabet_phones.index(p) for p in phones_wout_stress]
        stress = [p[-1] if p[-1] in '012' else 'NONE' for p in phones]
        stress_idxs = [self.phone_stress.index(s) for s in stress]
        if self.intra_word_pos:
            intra_word_positions = [t[4] for t in times_durs]
            word_lengths = [t[5] for t in times_durs]
        else:
            intra_word_positions = [0 for _ in phones]
            word_lengths = [1 for _ in phones]
        return wav, torch.tensor(sts), torch.tensor(ets), torch.tensor(durs), torch.tensor(phones_idxs), torch.tensor(stress_idxs), torch.tensor(intra_word_positions), torch.tensor(word_lengths)
        
class WordDurationDataset(DurationDataset):
    def __init__(self, mode, corpus_dir, meta_data_dir, glottal_kwargs, upstream_rate=160, use_spkr_embeds=False, **kwargs):
        super().__init__(mode, corpus_dir, meta_data_dir, glottal_kwargs, upstream_rate=upstream_rate, use_spkr_embeds=use_spkr_embeds, **kwargs)
        cache_path = os.path.join(CACHE_PATH, f"word/{self.mode}_times_durs.pkl")
        if os.path.isfile(cache_path):
            print(f"[WordDurationDataset] - loading cached {self.mode} times_durs from {cache_path}")
            with open(cache_path, "rb") as f:
                times_durs = pickle.load(f)
        else:
            times_durs = self.get_times_durs(self.dataset)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            print(f"[WordDurationDataset] - saving cached {self.mode} times_durs to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(times_durs, f)

        # filter out examples with duration less than min_frame_size
        self.times_durs = self.filter_short_durations(times_durs)

    def get_times_durs(self, dataset):
        if self.mode == "train":
            align_dir = os.path.join(self.corpus_dir, f"alignments/{self.mode}_clean_100")
        else:
            align_dir = os.path.join(self.corpus_dir, f"alignments/{self.mode}_clean")
        all_times_durs = []
        for item in tqdm.tqdm(dataset, desc=f"loading {self.mode} alignments", total=len(dataset)):
            align_path = os.path.join(
                align_dir,
                item+".TextGrid"
            )
            tg = textgrid.TextGrid.fromFile(align_path)
            all_times_durs.append(self.parse_words(tg))
        num_durs = [len(tds) for tds in all_times_durs]
        print(f"[WordDurationDataset] - {len(all_times_durs)} files, number of words: {min(num_durs)} ~ {max(num_durs)}, average: {sum(num_durs)/len(num_durs):.2f}")
        return all_times_durs

    def parse_words(self, tg):
        word_tier = tg.getFirst("words")
        times_words = [
            (interval.minTime, interval.maxTime, interval.maxTime-interval.minTime, interval.mark.strip().lower()) for interval in word_tier
            if interval.mark.strip() != ""
        ]
        return times_words
    
    def collate_fn(self, samples):
        return zip(*samples)

    def __getitem__(self, idx):
        path = self.dataset[idx]
        h5_file = self._get_h5_file()
        if h5_file is not None:
            wav = h5_file[path][:]
            # resample from 24000 to 16000
            wav = librosa.resample(wav.astype(np.float32), orig_sr=LIBRITTSR_SR, target_sr=SAMPLE_RATE)
            if self.return_glottal:
                wav = self.glottal_extractor.extract(torch.from_numpy(wav), idx)
        else:
            raise NotImplementedError("H5 file is required for loading wav data since LibriTTS-R audios have been converted to .h5 format.")
        
        times_durs = self.times_durs[idx]
        sts = [t[0] for t in times_durs]
        ets = [t[1] for t in times_durs]
        durs = [t[2] for t in times_durs]
        words = " ".join([t[3] for t in times_durs])
        words = words.replace("'", "").replace("_", "").replace("{", "").replace("}", "")

        return wav, torch.tensor(sts), torch.tensor(ets), torch.tensor(durs), words, None, None, None # token_ids instead of words
