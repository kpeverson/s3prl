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
from .dataset import TonesDataset
from argparse import Namespace
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

# class ContinuumCELoss(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         class_idxs = torch.arange(num_classes).unsqueeze(0)
#         cost_matrix = torch.abs(class_idxs - class_idxs.T).float()
#         self.soft_cost_matrix = torch.exp(-cost_matrix)
#         self.soft_cost_matrix = self.soft_cost_matrix / torch.sum(self.soft_cost_matrix, dim=-1, keepdim=True)  # Normalize the soft cost matrix

#     def forward(self, logits, targets):
#         log_probs = nn.functional.log_softmax(logits, dim=-1)
#         soft_targets = self.soft_cost_matrix.to(targets.device)[targets]
#         loss = -torch.sum(soft_targets * log_probs, dim=-1)
#         return loss.mean()

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.downstream = downstream_expert
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.expdir = expdir
        print(f"saving logs to {self.expdir}")

        root_dir = Path(self.datarc['file_path'])
        meta_data = self.datarc['meta_data']
        return_glottal = self.datarc.get('return_glottal', False)
        max_timestep = self.datarc.get('max_timestep', None)

        self.h5_path = self.datarc.get('h5_path', None)
        self.train_dataset = TonesDataset('train', root_dir, meta_data, max_timestep, return_glottal=return_glottal, sr=self.datarc.get('sr', 16000), h5_path=self.h5_path)
        self.dev_dataset = TonesDataset('dev', root_dir, meta_data, return_glottal=return_glottal, sr=self.datarc.get('sr', 16000), h5_path=self.h5_path)
        self.test_dataset = TonesDataset('test', root_dir, meta_data, return_glottal=return_glottal, sr=self.datarc.get('sr', 16000), h5_path=self.h5_path)

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.window_size = self.datarc.get('window_size', 0.2) # total window size in seconds
        self.register_buffer('best_score', torch.zeros(1))

        self.feature_rate = self.modelrc.get('feature_rate', 62.5)
        self.labels_mode = self.datarc.get('labels_mode', '0v1v2v3v4')
        assert all([digit in '01234v' for digit in self.labels_mode]), f"Invalid labels_mode {self.labels_mode}. Supported digits are 0, 1, 2, 3, 4 and 'v' as separator."
        assert all([digit in self.labels_mode for digit in '01234']), f"Labels mode {self.labels_mode} must contain all digits 0, 1, 2, 3, 4."
        self.labels_map = [-1] * 5
        for group_idx, group in enumerate(self.labels_mode.split('v')):
            for digit in group:
                assert digit in '01234', f"Invalid digit {digit} in labels_mode {self.labels_mode}. Supported digits are 0, 1, 2, 3, 4."
                self.labels_map[int(digit)] = group_idx
        # self.num_break_idxs = len(set(self.labels_map))
        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=1, # self.num_break_idxs,
            **model_conf,
        )

        # self.objective_mode = self.modelrc.get('objective', 'CrossEntropyLoss')
        # if self.objective_mode == 'ContinuumCELoss':
        #     self.objective = ContinuumCELoss(self.num_break_idxs)
        # else:
        # use binary cross entropy loss
        self.objective = nn.BCEWithLogitsLoss()

        self.save_metric = 'f1' # self.modelrc.get('save_metric', 'f1')
        # assert self.save_metric in ['f1', 'acc'], f"Unsupported save_metric {self.save_metric}. Supported metrics are 'f1' and 'acc'."

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
        return eval(f'self.get_{mode}_dataloader')()
    
    def convert_to_word_level(self, features, sts, ets):
        feats_list = []
        for fs, st, et in zip(features, sts, ets):
            for s, e in zip(st, et):
                s_idx = max(0, int(s * self.feature_rate))
                e_idx = min(len(fs), int(e * self.feature_rate))
                feats_list.append(fs[s_idx:e_idx])
        return feats_list
    
    def forward(self, mode, features, sts, ets, labels, h5_feats, filenames, records, **kwargs):
        device = features[0].device
        if self.h5_path is not None:
            # use h5_feats if available
            features = [h5_feat.to(device=device) for h5_feat in h5_feats]
        word_features = self.convert_to_word_level(features, sts, ets)
        word_features_len = torch.IntTensor(
            [len(feat) for feat in word_features]
        ).to(device=device)
        word_features = pad_sequence(word_features, batch_first=True)
        word_features = self.projector(word_features)
        predicted, _ = self.model(word_features, word_features_len)

        # combine all sub-lists of labels into a single list
        labels = torch.cat([torch.tensor(label, device=device) for label in labels])
        # map labels to the correct format
        labels = torch.LongTensor([self.labels_map[label.item()] for label in labels]).to(device)
        # print(f"predicted.squeeze(-1): type {predicted.squeeze(-1).dtype}, shape {predicted.squeeze(-1).shape}")
        # print(f"labels: type {labels.dtype}, shape {labels.shape}")
        loss = self.objective(predicted.squeeze(-1), labels.float())

        # get predicted class ids using threshold 0
        # print(f"predicted ({predicted.shape}): {predicted}")
        # exit()
        predicted_classid = (predicted > 0).long().squeeze(-1)
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records['filename'] += filenames
        records['predict_label'] += predicted_classid.cpu().tolist()
        records['truth_label'] += labels.cpu().tolist()

        return loss
    
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f'bu_radio_tones/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == "acc":
                    print(f"\n{mode} {key}: {average}")
                    f.write(f"{mode} at step {global_step}: {average}\n")
                    if mode == "dev" and average > self.best_score and self.save_metric == "acc":
                        self.best_score = torch.ones(1) * average
                        f.write(f"New best on {mode} at step {global_step}: {average}\n")
                        save_names.append(f"{mode}-best.ckpt")

        # Compute and log macro F1 score
        if len(records["predict_label"]) > 0 and len(records["truth_label"]) > 0:
            y_pred = np.array(records["predict_label"])
            y_true = np.array(records["truth_label"])
            f1 = f1_score(y_true, y_pred) #, average="macro")
            cm = confusion_matrix(y_true, y_pred, labels=[0,1])
            logger.add_scalar(
                f'bu_radio_tones/{mode}-f1',
                f1,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                print(f"\n{mode} confusion matrix:\n{cm}")
                print(f"\n{mode} F1: {f1}\n")
                f.write(f"{mode} F1 at step {global_step}: {f1}\n")
                # if mode == "dev" and f1 > self.best_score and self.save_metric == "f1":
                #     self.best_score = torch.ones(1) * f1
                #     f.write(f"New best on {mode} at step {global_step}: {f1}\n")
                #     save_names.append(f"{mode}-best.ckpt")

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
                lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_label"])]
                file.writelines(lines)
            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                lines = [f"{f} {t}\n" for f, t in zip(records["filename"], records["truth_label"])]
                file.writelines(lines)

        return save_names
