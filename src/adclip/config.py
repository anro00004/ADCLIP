import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"

MORGAN_FP_CSV = DATA_DIR / "substrates" / "morgan_fp_r4.csv"
REFERENCE_1AMU_FASTA = DATA_DIR / "alignment" / "1AMU.fasta"
DEFAULT_CORPUS_CSV = DATA_DIR / "corpus" / "preprocessed_single_label.csv"
DEFAULT_CORPUS_FASTA = DATA_DIR / "corpus" / "preprocessed_single_label.fasta"

with open(DATA_DIR / "model_config.json") as _f:
    _MODEL_CONFIG = json.load(_f)

POSITIONS = _MODEL_CONFIG["positions"]
ONEHOT_DIM = _MODEL_CONFIG["onehot_dim"]

MODEL_CTOR_ARGS = {
    "amino_acid_dim": _MODEL_CONFIG["amino_acid_dim"],
    "adomain_dims": _MODEL_CONFIG["adomain_dims"],
    "latent_dim": _MODEL_CONFIG["latent_dim"],
    "dropout": _MODEL_CONFIG["dropout"],
    "init_temperature": _MODEL_CONFIG["init_temperature"],
}


def checkpoint_path(name: str) -> Path:
    meta = _MODEL_CONFIG.get("checkpoints", {}).get(name)
    if meta is not None:
        path = DATA_DIR / meta["file"]
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint '{name}' is not available yet: {path} does not exist. "
                f"({meta.get('note', '')})"
            )
        return path
    path = Path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"'{name}' is neither a known checkpoint name (baseline, complete) "
            f"nor an existing file path."
        )
    return path 


def sidecar_meta_path(checkpoint_path) -> Path:
    return Path(checkpoint_path).with_suffix(".meta.json")


def checkpoint_meta(name: str) -> dict:
    meta = _MODEL_CONFIG.get("checkpoints", {}).get(name)
    if meta is None:
        sidecar = sidecar_meta_path(name)
        if sidecar.exists():
            with open(sidecar) as f:
                meta = json.load(f)
        else:
            raise KeyError(
                f"No metadata registered for checkpoint '{name}', and no sidecar "
                f"'{sidecar}' found next to it. Known checkpoints: "
                f"{list(_MODEL_CONFIG.get('checkpoints', {}))}."
            )
    if meta.get("desc_mean") is None:
        raise ValueError(f"Checkpoint '{name}' has no desc_mean/desc_std recorded yet.")
    return meta
