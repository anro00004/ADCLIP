from adclip import ADCLIP, corpus, features
import random

random.seed(0)
def test_query_by_substrate_default_corpus_ranks_true_domains_highly():
    df = corpus.load_default_corpus_df()
    substrate_counts = df['substrate'].value_counts()
    eligible_substrates = substrate_counts[substrate_counts >= 10].index.tolist()
    sampled_substrates = random.sample(eligible_substrates, 5)
    for target_substrate in sampled_substrates:
        smiles = df[df.substrate == target_substrate]["smiles"].iloc[0]
        true_ids = set(df[df.substrate == target_substrate]["new_domain_id"])

        model = ADCLIP.load(checkpoint="complete", device="cpu")
        result = model.query_substrate(smiles, top_k=20)

        assert set(result.columns) == {"query_id", "a_domain_id", "cosine", "rank", "unresolved_positions", "code", "code_idx"}
        top20_hit_rate = result["a_domain_id"].isin(true_ids).mean()
        assert top20_hit_rate > 0.5, (
            f"only {top20_hit_rate:.0%} of top-20 hits were true {target_substrate} domains")


def test_canonical_smiles_is_stable_under_reformatting(): 
    a = features.canonical_smiles("C1CCNC(C1)C(=O)O")
    b = features.canonical_smiles("OC(=O)C1CCCCN1")
    assert a == b
