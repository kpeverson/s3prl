import tempfile
from pathlib import Path
from typing import List

import torch

import s3prl
from s3prl.upstream.speakervec.speakervec_model import (
    SpeakervecConfig,
    SpeakervecModel,
    SpeakervecPretrainingConfig,
)
from s3prl.upstream.utils import load_fairseq_ckpt, merge_with_parent
from s3prl.util.download import _urls_to_filepaths


def load_and_convert_fairseq_ckpt(fairseq_source: str, output_path: str = None):
    from fairseq.data.dictionary import Dictionary

    state, cfg = load_fairseq_ckpt(fairseq_source)

    dicts: List[Dictionary] = state["task_state"]["dictionaries"]
    symbols = [dictionary.symbols for dictionary in dicts]

    output_state = {
        "task_cfg": cfg["task"],
        "model_cfg": cfg["model"],
        "model_weight": state["model"],
        "dictionaries_symbols": symbols,
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(output_state, output_path)

def fix_model_cfg(model_cfg, ckpt_wts, num_layers):
    """fix model_cfg to be compatible with SpeakervecModel"""
    # final dim is the first dim of the final_proj layer
    final_dim = ckpt_wts['final_proj.weight'].shape[0]
    model_cfg['final_dim'] = final_dim
    # change encoder_layers to num_layers
    model_cfg['encoder_layers'] = num_layers
    return model_cfg

def load_converted_model(ckpt: str):
    ckpt_state = torch.load(ckpt, map_location="cpu")
    ckpt_wts = ckpt_state['model']

    ckpt_state["cfg"]["model"] = fix_model_cfg(ckpt_state["cfg"]["model"], ckpt_wts, 12)

    task_cfg = merge_with_parent(SpeakervecPretrainingConfig, ckpt_state["cfg"]["task"]) # ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(SpeakervecConfig, ckpt_state["cfg"]["model"]) # ckpt_state["model_cfg"])
    
    model = SpeakervecModel(model_cfg, task_cfg, [None]) # ckpt_state["dictionaries_symbols"])

    model.load_state_dict(ckpt_wts, strict=False) # ckpt_state["model_weight"])
    return model, task_cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fairseq_ckpt")
    parser.add_argument(
        "--output_dir", default=Path(s3prl.__file__).parent.parent / "converted_ckpts"
    )
    args = parser.parse_args()

    Path(args.output_dir).parent.mkdir(exist_ok=True, parents=True)
    load_and_convert_fairseq_ckpt(
        args.fairseq_ckpt, Path(args.output_dir) / f"{Path(args.fairseq_ckpt).stem}.pt"
    )
