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
from .dataset import BreakIdxDataset
from argparse import Namespace
from pathlib import Path

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

        root_dir = Path(self.datarc['file_path'])
        meta_data = self.datarc['meta_data']
        return_glottal = self.datarc.get('return_glottal', False)

        self.train_dataset = BreakIdxDataset('train', root_dir, meta_data, self.datarc['max_timestep'], return_glottal=return_glottal, sr=self.datarc.get('sr', 16000))
        self.dev_dataset = BreakIdxDataset('dev', root_dir, meta_data, return_glottal=return_glottal, sr=self.datarc.get('sr', 16000))
        self.test_dataset = BreakIdxDataset('test', root_dir, meta_data, return_glottal=return_glottal, sr=self.datarc.get('sr', 16000))

        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.window_size = self.datarc.get('window_size', 0.2) # total window size in seconds
        self.objective = nn.CrossEntropyLoss()
        self.register_buffer('best_score', torch.zeros(1))

        self.feature_rate = self.modelrc.get('feature_rate', 62.5)
        self.labels_mode = self.datarc.get('labels_mode', '0v1v2v3v4')
        if self.labels_mode == '0v1v2v3v4':
            self.labels_map = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 4,
            }
        elif self.labels_mode == '012v3v4':
            self.labels_map = {
                0: 0,
                1: 0,
                2: 0,
                3: 1,
                4: 2,
            }
        else:
            raise ValueError(f"Unsupported labels_mode: {self.labels_mode}. Supported modes are '0v1v2v3v4' and '012v3v4'.")
        break_num = len(list(self.labels_map.values()))
        self.model = model_cls(
            input_dim=self.modelrc['projector_dim'],
            output_dim=break_num,
            **model_conf,
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
    
    def convert_to_boundary_level(self, features, times):
        feats_list = []
        for f, t in zip(features, times):
            st = t - self.window_size / 2
            et = t + self.window_size / 2
            st_idx = max(0, int(st * self.feature_rate))
            et_idx = min(len(f), int(et * self.feature_rate))
            feats_list.append(f[st_idx:et_idx])
        return feats_list
    
    def forward(self, mode, features, times, labels, filenames, records, **kwargs):
        device = features[0].device
        boundary_features = self.convert_to_boundary_level(features, times)
        boundary_features_len = torch.IntTensor(
            [len(feat) for feat in boundary_features]
        ).to(device=device)
        boundary_features = pad_sequence(boundary_features, batch_first=True)
        boundary_features = self.projector(boundary_features)
        predicted, _ = self.model(boundary_features, boundary_features_len)

        # combine all sub-lists of labels into a single list
        labels = torch.cat([torch.tensor(label, device=device) for label in labels])
        # map labels to the correct format
        labels = torch.IntTensor([self.labels_map[label.item()] for label in labels], device=device)
        loss = self.objective(predicted, labels)

        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records['filename'] += filenames
        records['predict_break'] += predicted_classid.cpu().tolist()
        records['truth_break'] += labels.cpu().tolist()

        return loss
    
    def log_records(self, mode, records, logger, global_step, **kwargs):
        save_names = []
        for key in ["acc", "loss"]:
            average = torch.FloatTensor(records[key]).mean().item()
            logger.add_scalar(
                f'bu_radio_breaks/{mode}-{key}',
                average,
                global_step=global_step
            )
            with open(Path(self.expdir) / "log.log", 'a') as f:
                if key == "acc":
                    print(f"{mode} {key}: {average}")
                    f.write(f"{mode} at step {global_step}: {average}\n")
                    if mode == "dev" and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f"New best on {mode} at step {global_step}: {average}\n")
                        save_names.append(f"{mode}-best.ckpt")

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_predict.txt", "w") as file:
                lines = [f"{f} {p}\n" for f, p in zip(records["filename"], records["predict_break"])]
                file.writelines(lines)
            with open(Path(self.expdir) / f"{mode}_truth.txt", "w") as file:
                lines = [f"{f} {t}\n" for f, t in zip(records["filename"], records["truth_break"])]
                file.writelines(lines)

        return save_names
