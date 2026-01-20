
import glob
import os

import textgrid
import tqdm

alignments_dir = "/gscratch/tial/data/LibriTTS-R/LibriTTS_R/alignments"
vocab_path = "/gscratch/tial/kpever/workspace/s3prl/s3prl/downstream/duration_libritts"

splits = ["dev_clean", "dev_other", "test_clean", "test_other", "train_clean_100", "train_clean_360"]

curr_vocab = set()

for split in splits:
    split_ali_dir = os.path.join(alignments_dir, split)
    ali_paths = glob.glob(os.path.join(split_ali_dir, "*.TextGrid"))
    for ali_path in tqdm.tqdm(ali_paths, desc=f"{split}"):
        tg = textgrid.TextGrid.fromFile(ali_path)
        phone_tier = tg.getFirst("phones")
        for interval in phone_tier:
            phone = interval.mark.strip()
            if phone != "":
                curr_vocab.add(phone)

with open(os.path.join(vocab_path, "phone_vocab.txt"), "w") as f:
    for phone in sorted(list(curr_vocab)):
        f.write(f"{phone}\n")