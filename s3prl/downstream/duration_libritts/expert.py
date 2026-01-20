###############
# IMPORTATION #
###############
import os
import math
import torch
import random
import pathlib
#-------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
#-------------#
from ..model import *
from .dataset import DurationDataset
from .utils import get_triphone_context, get_relative_intra_word_pos, NormalizedSinusoidalEncoding
from argparse import Namespace
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream_rate = upstream_rate
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir
        print(f"saving logs to {self.expdir}")

        self.text_only = self.modelrc.get('text_only', False)

        h5_dir = self.datarc['h5_dir']
        glottal_kwargs = self.datarc.get('glottal_kwargs', {"return_glottal": False})
        intra_word_pos = self.datarc.get('intra_word_pos', True)
        meta_data_dir = self.datarc['meta_data_dir']
        vocab_path = self.datarc['vocab_path']

        self.dur_mode = self.datarc.get('dur_mode', 'syl')
        if self.dur_mode != 'phone':
            raise NotImplementedError("Word-level duration prediction is not implemented yet.")

        self.train_dataset = DurationDataset('train', h5_dir, meta_data_dir, vocab_path, glottal_kwargs, dur_mode=self.dur_mode, intra_word_pos=intra_word_pos, upstream_rate=upstream_rate)
        self.dev_dataset = DurationDataset('dev', h5_dir, meta_data_dir, vocab_path, glottal_kwargs, dur_mode=self.dur_mode, intra_word_pos=intra_word_pos, upstream_rate=upstream_rate)
        self.test_dataset = DurationDataset('test', h5_dir, meta_data_dir, vocab_path, glottal_kwargs, dur_mode=self.dur_mode, intra_word_pos=intra_word_pos, upstream_rate=upstream_rate)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.register_buffer('best_loss', torch.zeros(1))

        self.phoneme_dim = self.modelrc.get('phoneme_dim', 256)

        print(f"[Downstream Expert] Upstream dimension: {upstream_dim}")

        self.feature_rate = self.modelrc.get('feature_rate', 62.5)

        if self.text_only:
            model_input_dim = self.phoneme_dim*3
        else:
            model_input_dim = self.modelrc['projector_dim'] + self.phoneme_dim*3
            self.dur_mode_pooler = eval(self.modelrc.get('dur_mode_pooling', 'AttentivePooling'))(
                input_dim=self.modelrc['projector_dim'],
                activation=self.modelrc.get('pooling_activation', 'ReLU')
            )
        self.model = model_cls(
            input_dim=model_input_dim,
            output_dim=1,
            **model_conf,
        )
        self.phone_embedding = nn.Embedding(len(self.train_dataset.arpabet_phones), self.phoneme_dim)
        self.stress_embedding = nn.Embedding(len(self.train_dataset.phone_stress), self.phoneme_dim)
        self.intra_word_pos_enc = NormalizedSinusoidalEncoding(self.phoneme_dim)

        self.objective = nn.MSELoss()
        self.save_metric = 'mse'

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn, sampler=sampler
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)
    
    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader()')
    
    def convert_to_dur_mode_level(self, features, sts, ets):
        device = features[0].device
        feats_list = []
        for fs, st, et in zip(features, sts, ets):
            for s, e in zip(st, et):
                s_idx = max(0, int(s * self.feature_rate))
                e_idx = min(len(fs), int(e * self.feature_rate))
                feats_list.append(fs[s_idx:e_idx])
        dur_mode_feats_lens = torch.IntTensor(
            [len(feat) for feat in feats_list]
        ).to(device=device)
        dur_mode_feats = pad_sequence(feats_list, batch_first=True)
        return dur_mode_feats, dur_mode_feats_lens

    # def get_triphone_context(self, phone_idxs, stress_idxs, intra_word_positions):
    #     """
    #     input:
    #         phone_idxs: list of Tensor, each Tensor is of shape (num_phones,)
    #         stress_idxs: list of Tensor, each Tensor is of shape (num_phones,)
    #         intra_word_positions: list of Tensor, each Tensor is of shape (num_phones,)
    #     returns:
    #         triphone_phone_idxs: list of Tensor, each Tensor is of shape (num_phones, 3)
    #         triphone_stress_idxs: list of Tensor, each Tensor is of shape (num_phones, 3)
    #         triphone_intra_word_positions: list of Tensor, each Tensor is of shape (num_phones, 3)

    #     triphone_phone_idxs contains the [prev_phone_idx, phone_idx, next_phone_idx] for each phone
    #     """
    #     triphone_phone_idxs = []
    #     triphone_stress_idxs = []
    #     triphone_intra_word_positions = []
    #     for p_idxs, s_idxs, iwp_idxs in zip(phone_idxs, stress_idxs, intra_word_positions):
    #         num_phones = len(p_idxs)
    #         # use -1 for padding
    #         triphone_p_idxs = torch.stack([
    #             torch.cat([torch.tensor([-1], device=p_idxs.device), p_idxs[:-1]]), # previous phone
    #             p_idxs, # current phone
    #             torch.cat([p_idxs[1:], torch.tensor([-1], device=p_idxs.device)]) # next phone
    #         ], dim=-1)
    #         triphone_phone_idxs.append(triphone_p_idxs)
    #         triphone_s_idxs = torch.stack([
    #             torch.cat([torch.tensor([-1], device=s_idxs.device), s_idxs[:-1]]), # previous stress
    #             s_idxs, # current stress
    #             torch.cat([s_idxs[1:], torch.tensor([-1], device=s_idxs.device)]) # next stress
    #         ], dim=-1)
    #         triphone_stress_idxs.append(triphone_s_idxs)
    #         triphone_iwp = torch.stack([
    #             torch.cat([torch.tensor([-1], device=p_idxs.device), iwp_idxs[:-1]]), # previous intra word position
    #             iwp_idxs, # current intra word position
    #             torch.cat([iwp_idxs[1:], torch.tensor([-1], device=p_idxs.device)]) # next intra word position
    #         ], dim=-1)
    #         triphone_intra_word_positions.append(triphone_iwp)

    #     return triphone_phone_idxs, triphone_stress_idxs, triphone_intra_word_positions
    
    # def get_relative_intra_word_pos(self, intra_word_positions, word_lengths):
    #     """
    #     input:
    #         intra_word_positions: list of Tensor, each Tensor of shape (num_phones,)
    #         word_lengths: list of Tensor, each Tensor of shape (num_phones,)
    #     returns:
    #         relative_intra_word_pos: list of Tensor, each Tensor of shape (num_phones,)
    #     """
    #     relative_intra_word_positions = []
    #     for iwp, wl in zip(intra_word_positions, word_lengths):
    #         relative_iwp = iwp.float() / wl.float()
    #         relative_intra_word_positions.append(relative_iwp)
    #     return relative_intra_word_positions
        
    def forward(self, mode, features, sts, ets, durs, phone_idxs, stress_idxs, intra_word_positions, word_lengths, records, **kwargs):
        device = features[0].device

        all_durs = torch.cat([d for d in durs]).to(device=device).unsqueeze(-1)

        rel_intra_word_positions = get_relative_intra_word_pos(intra_word_positions, word_lengths)
        triphone_phone_idxs, triphone_stress_idxs, triphone_intra_word_positions = get_triphone_context(phone_idxs, stress_idxs, rel_intra_word_positions)
        all_phone_idxs = torch.cat([ph for ph in triphone_phone_idxs]).to(device=device) # shape (total_num_phones, 3)
        all_stress_idxs = torch.cat([st for st in triphone_stress_idxs]).to(device=device) # shape (total_num_phones, 3)
        all_rel_iwp = torch.cat([iwp for iwp in triphone_intra_word_positions]).to(device=device) # shape (total_num_phones, 3)
        # get phone and stress embeddings (each of shape (total_num_phones, 3, phoneme_dim))
        # map padding index -1 to zero for now - will zero out the embeddings later
        phone_embeds = self.phone_embedding(torch.clamp(all_phone_idxs, min=0))
        stress_embeds = self.stress_embedding(torch.clamp(all_stress_idxs, min=0))
        intra_word_pos_enc = self.intra_word_pos_enc(all_rel_iwp)  # shape (total_num_phones, 3, phoneme_dim)
        # zero out the embeddings for padding index -1
        phone_embeds = phone_embeds * (all_phone_idxs.unsqueeze(-1) != -1).float()
        stress_embeds = stress_embeds * (all_stress_idxs.unsqueeze(-1) != -1).float()
        intra_word_pos_enc = intra_word_pos_enc * (all_rel_iwp.unsqueeze(-1) != -1).float()
        # add all three
        phone_embeds += stress_embeds + intra_word_pos_enc
        # concatenate on last two dimensions, e.g. (total_num_phones, 3, phoneme_dim) -> (total_num_phones, 3*phoneme_dim)
        phone_embeds = phone_embeds.view(phone_embeds.size(0), -1)

        if self.text_only:
            dur_mode_features = phone_embeds
        else:
            # concatenate dur_mode_features with phone_embeds along last dimension
            dur_mode_features, dur_mode_feats_lens = self.convert_to_dur_mode_level(features, sts, ets)
            dur_mode_features = self.projector(dur_mode_features)
            dur_mode_features, _ = self.dur_mode_pooler(dur_mode_features, dur_mode_feats_lens)
            dur_mode_features = torch.cat([dur_mode_features, phone_embeds], dim=-1)

        # check dur_mode_features for nans
        if torch.isnan(dur_mode_features).any():
            print("dur_mode_features contains NaNs")
            exit(1)
        predicted_durs, _ = self.model(dur_mode_features)
        # check predicted_durs for nans
        if torch.isnan(predicted_durs).any():
            print("predicted_durs contains NaNs")
            exit(1)
        # check all_durs for nans
        if torch.isnan(all_durs).any():
            print("all_durs contains NaNs")
            exit(1)
        loss = self.objective(predicted_durs, all_durs)
        # check loss for nans
        if torch.isnan(loss).any():
            print("loss is NaN")
            exit(1)
        # print(f"loss: {loss.item()}")

        records['loss'].append(loss.item())

        return loss
    
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        keys = ["loss"]
        for key in keys:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f"{self.dur_mode}_duration_libritts/{mode}-{key}",
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", "a") as f:
                if key == "loss":
                    print(f"{mode} {key} at step {global_step}: {average}")
                    f.write(f"{mode} {key} at step {global_step}: {average}")
                    if mode == "dev" and average < self.best_loss:
                        self.best_loss = torch.ones(1) * average
                        f.write(f"New best {key} at step {global_step}: {average}\n")
                        save_names.append(f"{mode}-best.ckpt")
        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_loss.txt", "w") as f:
                lines = [f"{x}\n" for x in records["loss"]]
                f.writelines(lines)
        
        return save_names
