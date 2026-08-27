import tempfile
from pathlib import Path
from adclip import ADCLIP, alignment, corpus


def test_query_adomain_ranks_true_substrate_highly():
    df = corpus.load_default_corpus_df()
    sample = df.sample(n=10, random_state=0).reset_index(drop=True)
    pool_rows, query_row = sample.iloc[:9], sample.iloc[9]

    with tempfile.TemporaryDirectory() as tmp:
        pool_path = Path(tmp) / "pool.fasta" 
        alignment.write_fasta(
            {r.new_domain_id: r.a_domain_sequence for r in pool_rows.itertuples()}, pool_path)

        model = ADCLIP.load(checkpoint="complete", device="cpu")
        result = model.query_adomain(
            {query_row.new_domain_id: query_row.a_domain_sequence}, pool=str(pool_path), top_k=5)

    assert set(result.columns) == {"query_id", "substrate", "cosine", "rank", "unresolved_positions", "code", "code_idx"}
    assert (result["cosine"] >= -1.0001).all() and (result["cosine"] <= 1.0001).all()
    assert list(result["rank"]) == [1, 2, 3, 4, 5]
    top3_substrates = set(result.iloc[:3]["substrate"])
    assert query_row.substrate in top3_substrates, (
        f"expected '{query_row.substrate}' in top-3, got {list(top3_substrates)}")
