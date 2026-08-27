"""Substrate side: the bundled fingerprint table (default bank for
query_adomain) and fresh SMILES -> fingerprint featurization (required
input for query_substrate, and for brand-new substrates in CPT).
"""
import numpy as np
import pandas as pd
import torch

from . import config, features


def load_fingerprints():
    df = pd.read_csv(config.MORGAN_FP_CSV)
    bit_cols = [col for col in df.columns if col != "substrate"]
    return df["substrate"].tolist(), df[bit_cols].values.astype(np.float32)


def default_substrates_latents(model, device):
    names, fps = load_fingerprints()
    model.eval()
    with torch.no_grad():
        latents = model.project_aa_latent(torch.tensor(fps, dtype=torch.float32).to(device))
    return names, latents.detach().cpu().numpy().astype("float32")


def custom_substrates_latents(model, name_to_smiles: dict, device):
    names = list(name_to_smiles)
    fps = np.stack([features.compute_morgan_fingerprint(name_to_smiles[n]) for n in names])
    model.eval()
    with torch.no_grad():
        latents = model.project_aa_latent(torch.tensor(fps, dtype=torch.float32).to(device))
    return names, latents.detach().cpu().numpy().astype("float32")


def generate_latent_from_smiles(model, smiles: str, device):
    fp = features.compute_morgan_fingerprint(smiles)
    model.eval()
    with torch.no_grad():
        latent = model.project_aa_latent(torch.tensor(fp, dtype=torch.float32).to(device))
    return latent.detach().cpu().numpy().astype("float32")
