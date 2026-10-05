"""
Precomputed-alignment check: align once with MUSCLE against a small pool,
save the alignment, then pass it back as aligned_fasta and confirm the
result is identical without rerunning MUSCLE. Also covers the error cases.

"""

import tempfile
from pathlib import Path
import pandas as pd
import pytest
from adclip import ADCLIP, alignment, config, corpus


def _small_pool_and_queries(tmp):
    df = corpus.load_default_corpus_df()
    sample = df.sample(n=10, random_state=0).reset_index(drop=True)
    pool_rows, query_rows = sample.iloc[:8], sample.iloc[8:]

    pool_path = Path(tmp) / "pool.fasta"
    alignment.write_fasta(
        {r.new_domain_id: r.a_domain_sequence for r in pool_rows.itertuples()}, pool_path)
    queries = {r.new_domain_id: r.a_domain_sequence for r in query_rows.itertuples()}
    return pool_path, queries


def test_saved_alignment_roundtrip_matches_muscle():
    with tempfile.TemporaryDirectory() as tmp:
        pool_path, queries = _small_pool_and_queries(tmp)
        aln_path = Path(tmp) / "aln.fasta"

        fresh = alignment.align_new_sequences(
            queries, pool=str(pool_path), threads=2, verbose=False, save_alignment_path=aln_path)

        assert aln_path.exists()
        saved = alignment.read_fasta(aln_path)
        (ref_id,) = alignment.read_fasta(config.REFERENCE_1AMU_FASTA)
        assert ref_id in saved
        assert set(queries) <= set(saved)

        reused = alignment.align_new_sequences(
            queries, pool=str(pool_path), verbose=False, aligned_fasta=aln_path)

    assert reused == fresh


def test_aligned_fasta_and_save_path_together_raise():
    with pytest.raises(ValueError):
        alignment.align_new_sequences(
            {"q": "MSTA"}, verbose=False, aligned_fasta="a.fasta", save_alignment_path="b.fasta")


def test_alignment_without_reference_raises():
    with tempfile.TemporaryDirectory() as tmp:
        aln_path = Path(tmp) / "no_ref.fasta"
        alignment.write_fasta({"q": "MS-TA"}, aln_path)
        with pytest.raises(alignment.AlignmentError, match="1AMU"):
            alignment.align_new_sequences({"q": "MSTA"}, verbose=False, aligned_fasta=aln_path)


def test_alignment_missing_query_id_raises():
    with tempfile.TemporaryDirectory() as tmp:
        ref_records = alignment.read_fasta(config.REFERENCE_1AMU_FASTA)
        aln_path = Path(tmp) / "no_query.fasta"
        alignment.write_fasta(ref_records, aln_path)
        with pytest.raises(alignment.AlignmentError, match="not found in the alignment"):
            alignment.align_new_sequences({"q": "MSTA"}, verbose=False, aligned_fasta=aln_path)


def test_query_substrate_rejects_alignment_args_without_corpus_fasta():
    # the guard runs before the model is touched, so no checkpoint is needed
    model = ADCLIP(model=None, device="cpu", checkpoint_name="unused")
    with pytest.raises(ValueError, match="corpus_fasta"):
        model.query_substrate("C1CCNC(C1)C(=O)O", aligned_fasta="aln.fasta")
    with pytest.raises(ValueError, match="corpus_fasta"):
        model.query_substrate("C1CCNC(C1)C(=O)O", save_alignment_path="aln.fasta")


def test_query_adomain_with_saved_alignment_matches_fresh_run():
    with tempfile.TemporaryDirectory() as tmp:
        pool_path, queries = _small_pool_and_queries(tmp)
        aln_path = Path(tmp) / "aln.fasta"

        model = ADCLIP.load(checkpoint="complete", device="cpu")
        fresh = model.query_adomain(queries, pool=str(pool_path), threads=2, top_k=5,
                                    save_alignment_path=aln_path)
        reused = model.query_adomain(queries, top_k=5, aligned_fasta=aln_path)

    pd.testing.assert_frame_equal(reused, fresh)
