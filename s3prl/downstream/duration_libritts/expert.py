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
from .dataset import DurationDataset, PhoneDurationDataset, WordDurationDataset
from .utils import get_triphone_context, get_relative_intra_word_pos, NormalizedSinusoidalEncoding
from argparse import Namespace
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModel, AutoTokenizer
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

        self.dur_mode = self.datarc.get('dur_mode', 'syl')
        if self.dur_mode not in ['phone', 'word']:
            raise NotImplementedError("Word-level duration prediction is not implemented yet.")
        if self.dur_mode == 'phone':
            self.phoneme_dim = self.modelrc.get('phoneme_dim', 256)
            vocab_path = self.datarc['vocab_path']
            self.train_dataset = PhoneDurationDataset('train', h5_dir, meta_data_dir, vocab_path, glottal_kwargs, intra_word_pos=intra_word_pos, upstream_rate=upstream_rate)
            self.dev_dataset = PhoneDurationDataset('dev', h5_dir, meta_data_dir, vocab_path, glottal_kwargs, intra_word_pos=intra_word_pos, upstream_rate=upstream_rate)
            self.test_dataset = PhoneDurationDataset('test', h5_dir, meta_data_dir, vocab_path, glottal_kwargs, intra_word_pos=intra_word_pos, upstream_rate=upstream_rate)
        if self.dur_mode == 'word':
            text_model = self.datarc.get('text_model', 'distilbert-base-uncased')
            self.bert_model = AutoModel.from_pretrained(text_model)
            self.tokenizer = AutoTokenizer.from_pretrained(text_model)
            self.train_dataset = WordDurationDataset('train', h5_dir, meta_data_dir, glottal_kwargs, upstream_rate=upstream_rate)
            self.dev_dataset = WordDurationDataset('dev', h5_dir, meta_data_dir, glottal_kwargs, upstream_rate=upstream_rate)
            self.test_dataset = WordDurationDataset('test', h5_dir, meta_data_dir, glottal_kwargs, upstream_rate=upstream_rate)
        
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.register_buffer('best_loss', torch.zeros(1))

        print(f"[Downstream Expert] Upstream dimension: {upstream_dim}")

        self.feature_rate = self.modelrc.get('feature_rate', 62.5)

        if self.text_only:
            if self.dur_mode == 'phone':
                model_input_dim = self.phoneme_dim*3
            elif self.dur_mode == 'word':
                model_input_dim = self.bert_model.config.hidden_size
        else:
            if self.dur_mode == 'phone':
                model_input_dim = self.modelrc['projector_dim'] + self.phoneme_dim*3
                self.phone_embedding = nn.Embedding(len(self.train_dataset.arpabet_phones), self.phoneme_dim)
                self.stress_embedding = nn.Embedding(len(self.train_dataset.phone_stress), self.phoneme_dim)
                self.intra_word_pos_enc = NormalizedSinusoidalEncoding(self.phoneme_dim)
            elif self.dur_mode == 'word':
                # add text model hidden size + projector dim
                model_input_dim = self.modelrc['projector_dim'] + self.bert_model.config.hidden_size

        self.dur_mode_pooler = eval(self.modelrc.get('dur_mode_pooling', 'AttentivePooling'))(
            input_dim=self.modelrc['projector_dim'],
            activation=self.modelrc.get('pooling_activation', 'ReLU')
        )
        self.model = model_cls(
            input_dim=model_input_dim,
            output_dim=1,
            **model_conf,
        )

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

    def convert_token2word_embeds(self, token_embeds, token2word_idxs):

        device = token_embeds.device
        B, T, D = token_embeds.shape

        flat_embeds = token_embeds.reshape(-1, D) # (B*T, D)

        valid = token2word_idxs >= 0  # (B, T)
        num_words_per_sent = token2word_idxs.max(dim=1).values + 1
        offsets = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            num_words_per_sent.cumsum(0)[:-1]
        ])
        global_token2word_idxs = torch.where(
            valid,
            token2word_idxs + offsets[:, None],
            token2word_idxs,    # -1 stays -1
        )
        flat_token2word_idxs = global_token2word_idxs.reshape(-1)  # (B*T,)

        # keep only valid tokens
        valid = flat_token2word_idxs >= 0  # (num_tokens,)
        flat_token2word_idxs = flat_token2word_idxs[valid]  # (num_tokens,)
        flat_embeds = flat_embeds[valid] # (num_tokens, D)

        # aggregate token embeddings to word embeddings (mean numerator)
        total_words = num_words_per_sent.sum().item()
        word_embeds = torch.zeros(total_words, D, device=device)  # (total_words, D)
        word_embeds.index_add_(0, flat_token2word_idxs, flat_embeds)

        # get counts of tokens in each word (mean denominator)
        token_in_word_counts = torch.zeros(total_words, device=device)  # (total_words,)
        token_in_word_counts.index_add_(
            0, flat_token2word_idxs, torch.ones_like(flat_token2word_idxs, dtype=torch.float, device=device)
        )

        # mean
        word_embeds = word_embeds / token_in_word_counts.unsqueeze(-1)

        return word_embeds
        
    def forward(self, mode, features, sts, ets, durs, text_input, stress_ids, intra_word_positions, word_lengths, records, **kwargs):
        device = features[0].device

        all_durs = torch.cat([d for d in durs]).to(device=device).unsqueeze(-1)

        if self.dur_mode == 'phone':
            # text input: phone indices
            rel_intra_word_positions = get_relative_intra_word_pos(intra_word_positions, word_lengths)
            triphone_phone_idxs, triphone_stress_ids, triphone_intra_word_positions = get_triphone_context(text_input, stress_ids, rel_intra_word_positions)
            all_phone_idxs = torch.cat([ph for ph in triphone_phone_idxs]).to(device=device) # shape (total_num_phones, 3)
            all_stress_ids = torch.cat([st for st in triphone_stress_ids]).to(device=device) # shape (total_num_phones, 3)
            all_rel_iwp = torch.cat([iwp for iwp in triphone_intra_word_positions]).to(device=device) # shape (total_num_phones, 3)
            # get phone and stress embeddings (each of shape (total_num_phones, 3, phoneme_dim))
            # map padding index -1 to zero for now - will zero out the embeddings later
            phone_embeds = self.phone_embedding(torch.clamp(all_phone_idxs, min=0))
            stress_embeds = self.stress_embedding(torch.clamp(all_stress_ids, min=0))
            intra_word_pos_enc = self.intra_word_pos_enc(all_rel_iwp)  # shape (total_num_phones, 3, phoneme_dim)
            # zero out the embeddings for padding index -1
            phone_embeds = phone_embeds * (all_phone_idxs.unsqueeze(-1) != -1).float()
            stress_embeds = stress_embeds * (all_stress_ids.unsqueeze(-1) != -1).float()
            intra_word_pos_enc = intra_word_pos_enc * (all_rel_iwp.unsqueeze(-1) != -1).float()
            # add all three
            phone_embeds += stress_embeds + intra_word_pos_enc
            # concatenate on last two dimensions, e.g. (total_num_phones, 3, phoneme_dim) -> (total_num_phones, 3*phoneme_dim)
            text_embeds = phone_embeds.view(phone_embeds.size(0), -1)
        elif self.dur_mode == 'word':
            # text_input: sentences (list of strings)
            with torch.no_grad():
                tokenizer_output = self.tokenizer(
                    text_input,
                    return_tensors='pt',
                    return_offsets_mapping=True,
                    padding=True,
                    add_special_tokens=True,
                )
                token_ids = tokenizer_output['input_ids'].to(device=device)  # shape (batch_size, seq_len)
                token2word_idxs = []
                for i in range(len(text_input)):
                    word_ids = tokenizer_output.word_ids(batch_index=i)  # list of length seq_len
                    word_ids = torch.LongTensor([idx if idx is not None else -1 for idx in word_ids])  # convert None to -1
                    token2word_idxs.append(word_ids)
                token2word_idxs = torch.stack(token2word_idxs, dim=0).to(device=device)  # shape (batch_size, seq_len)
                attention_mask = (token_ids != self.tokenizer.pad_token_id).to(device=device)
                token_embeds = self.bert_model(
                    input_ids=token_ids,
                    attention_mask=attention_mask
                ).last_hidden_state  # shape (batch_size, max_num_words, hidden_size)
                # get word-level embeds from token-level embeds
                text_embeds = self.convert_token2word_embeds(token_embeds, token2word_idxs)  # shape (total_num_words, hidden_size)
        if self.text_only:
            dur_mode_features = text_embeds
        else:
            # concatenate dur_mode_features with text_embeds along last dimension
            dur_mode_features, dur_mode_feats_lens = self.convert_to_dur_mode_level(features, sts, ets)
            dur_mode_features = self.projector(dur_mode_features)
            dur_mode_features, _ = self.dur_mode_pooler(dur_mode_features, dur_mode_feats_lens)
            dur_mode_features = torch.cat([dur_mode_features, text_embeds], dim=-1)
        predicted_durs, _ = self.model(dur_mode_features)
        loss = self.objective(predicted_durs, all_durs)

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
