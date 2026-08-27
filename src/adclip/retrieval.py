
import numpy as np
import pandas as pd

from . import corpus as corpus_module


def rank_against_pool(query_id, query_latent, bank_ids, bank_latents,
                       target_col: str, top_k=None, unresolved_positions=None,
                       code=None, code_idx=None) -> pd.DataFrame:
    index = corpus_module.FaissCorpusIndex(bank_ids, bank_latents)
    k = top_k if top_k is not None else len(bank_ids)
    hits = index.search(query_latent, top_k=k)
    rows = {
        "query_id": [query_id] * len(hits),
        target_col: [hid for hid, _ in hits],
        "cosine": [score for _, score in hits],
        "rank": np.arange(1, len(hits) + 1),
    }
    if unresolved_positions is not None:
        rows["unresolved_positions"] = [unresolved_positions] * len(hits)
    if code is not None:
        rows["code"] = [code] * len(hits)
    if code_idx is not None:
        rows["code_idx"] = [code_idx] * len(hits)
    return pd.DataFrame(rows)
