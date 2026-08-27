""" Depends on --run_full_corpus.
    Realigns against the full dataset. Skipped by default:

    pytest tests/test_alignment_pool_training.py --run_full_corpus 
"""
import pytest
from adclip import corpus, alignment


@pytest.mark.skipif("not config.getoption('--run_full_corpus')")
def test_training_pool_matches_static_mapping():
    df = corpus.load_default_corpus_df()
    samples_df = df.sample(n=50, random_state=0)
    passes = 0
    for idx in range(len(samples_df)):
        query_row = samples_df.iloc[idx]

        result = alignment.align_new_sequences(
            {query_row.new_domain_id: query_row.a_domain_sequence}, pool="training", threads=8)

        recovered = result[query_row.new_domain_id]["code_idx"]
        truth = query_row.code_idx
        if recovered == truth:
            passes += 1

    assert passes >= round(0.75 * len(samples_df)), (
        f"only {passes}/{len(samples_df)} sequences recovered an exact code_idx match")
