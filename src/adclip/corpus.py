
import ast
import faiss
import numpy as np
import pandas as pd
from . import alignment, config, features


def load_default_corpus_df() -> pd.DataFrame:
    df = pd.read_csv(config.DEFAULT_CORPUS_CSV)
    df["code_idx"] = df["code_idx"].apply(ast.literal_eval)
    return df


def _onehot_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.stack([
        features.build_adomain_onehot(row.a_domain_sequence, row.code_idx)
        for row in df.itertuples()
    ])


def build_latent_adomains_from_df(model, df: pd.DataFrame, device, id_col="new_domain_id"):
    ad_emb = _onehot_matrix(df)
    latents = model.project_adomain_latent(ad_emb, device=device)
    code_map = {row.__getattribute__(id_col): features.code_idx_to_residues(row.a_domain_sequence, row.code_idx)
                for row in df.itertuples()}
    code_idx_map = {row.__getattribute__(id_col): row.code_idx for row in df.itertuples()}
    return df[id_col].tolist(), latents, code_map, code_idx_map


def build_latent_adomains_from_fasta(model, fasta_path, device, pool="training", threads=4):
    raw = alignment.read_fasta(fasta_path)
    aligned_info = alignment.align_new_sequences(raw, pool=pool, threads=threads)
    ids = list(raw)
    ad_emb = np.stack([
        features.build_adomain_onehot(raw[i], aligned_info[i]["code_idx"]) for i in ids
    ])
    latents = model.project_adomain_latent(ad_emb, device=device)
    unresolved_map = {i: aligned_info[i]["unresolved_positions"] for i in ids}
    code_map = {i: features.code_idx_to_residues(raw[i], aligned_info[i]["code_idx"]) for i in ids}
    code_idx_map = {i: aligned_info[i]["code_idx"] for i in ids}
    return ids, latents, unresolved_map, code_map, code_idx_map


class FaissCorpusIndex:


    def __init__(self, ids, latents):
        self.ids = ids
        self.latents = np.asarray(latents, dtype="float32")
        self.index = faiss.IndexFlatIP(self.latents.shape[1])
        self.index.add(self.latents)

    def search(self, query_latent, top_k=None):
        k = top_k or len(self.ids)
        q = np.asarray(query_latent, dtype="float32").reshape(1, -1)
        D, I = self.index.search(q, k)
        return [(self.ids[i], float(d)) for i, d in zip(I[0], D[0]) if i != -1]
