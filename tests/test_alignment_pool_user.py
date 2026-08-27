"""
Fast alignment check: pulls a few known sequences out of the
corpus, realign one of them against a small synthetic pool of the others,
and confirm the recovered code_idx matches.

"""

import tempfile
from pathlib import Path
from adclip import alignment, corpus


def test_small_pool_recovers_known_code_idx():
    df = corpus.load_default_corpus_df()
    sample = df.sample(n=10, random_state=0).reset_index(drop=True)
    pool_rows, query_row = sample.iloc[:9], sample.iloc[9]

    with tempfile.TemporaryDirectory() as tmp:
        pool_path = Path(tmp) / "pool.fasta"
        alignment.write_fasta(
            {r.new_domain_id: r.a_domain_sequence for r in pool_rows.itertuples()}, pool_path)

        result = alignment.align_new_sequences(
            {query_row.new_domain_id: query_row.a_domain_sequence},
            pool=str(pool_path), threads=2, verbose=False)

    recovered = result[query_row.new_domain_id]["code_idx"]
    truth = query_row.code_idx
    n_match = sum(a == b for a, b in zip(recovered, truth))

    assert len(recovered) == 16
    assert n_match >= 14, f"only {n_match}/16 anchors matched: recovered={recovered} truth={truth}"
