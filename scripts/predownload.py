"""Pre-cache models + preference dataset on an internet-connected node.

Compute nodes on coe-hpc3 are air-gapped, so run this once on the login node
(coe-hpc1) to populate the shared ~/.cache/huggingface; jobs then run with
HF_HUB_OFFLINE=1 / HF_DATASETS_OFFLINE=1.

Note: the login node's old glibc can't build `datasets`/`pyarrow`, so we fetch the
dataset *repo files* with huggingface_hub (pure Python) rather than load_dataset;
the compute-side loader (rlhf.data._load_split) reads those parquet shards directly.

    python scripts/predownload.py
"""
from huggingface_hub import snapshot_download

MODELS = ["gpt2", "gpt2-medium"]
DATASETS = ["Dahoas/rm-static"]

for m in MODELS:
    p = snapshot_download(m, allow_patterns=["*.json", "*.txt", "*.safetensors",
                                             "merges.txt", "vocab.json", "*.model"])
    print("MODEL_OK", m, p)

for d in DATASETS:
    p = snapshot_download(repo_id=d, repo_type="dataset")
    print("DATASET_OK", d, p)

print("PREDOWNLOAD_DONE")
