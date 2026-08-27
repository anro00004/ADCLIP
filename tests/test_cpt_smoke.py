import tempfile
from pathlib import Path

import pandas as pd

from adclip import ADCLIP, corpus

HELD_OUT_SUBSTRATES = [
    "p-hydroxyphenylglycine",
    "piperazic_acid",
    "p-aminobenzoic_acid",
    "2-aminoadipic_acid",
    "pipecolic_acid",
]
N_EXAMPLES_PER_SUBSTRATE = 5
N_SEEDS = 10
MIN_FRACTION_IMPROVED = 0.9
MIN_MEAN_DELTA = 0.1


def test_cpt_smoke_teaches_held_out_substrates():
    """
    
    All 5 held-out substrates the baseline checkpoint never trained on
    should get noticeably more similar to their own A-domains after a few
    CPT epochs. The test passes if majority reliably do.
    
    """
    df = corpus.load_default_corpus_df()

    deltas = []
    for seed in range(N_SEEDS):
        sub_rows = pd.concat([
            df[df.substrate == s].sample(
                n=min(N_EXAMPLES_PER_SUBSTRATE, (df.substrate == s).sum()), random_state=seed)
            for s in HELD_OUT_SUBSTRATES
        ])

        with tempfile.TemporaryDirectory() as tmp:
            pairs_path = Path(tmp) / "pairs.csv"
            sub_rows[["a_domain_sequence", "smiles"]].to_csv(pairs_path, index=False)

            pool_sample = df.sample(n=10, random_state=seed)
            pool_path = Path(tmp) / "pool.fasta"
            with open(pool_path, "w") as f:
                for r in pool_sample.itertuples():
                    f.write(f">{r.new_domain_id}\n{r.a_domain_sequence}\n")

            model = ADCLIP.load(checkpoint="baseline", device="cpu")
            model.continue_pretrain(
                str(pairs_path), lr=1e-4, epochs=10, patience=5, l_atp=0.2, l_prop=0.3,
                batch_size=64, pool=str(pool_path), out_checkpoint=str(Path(tmp) / "out.pt"))

            report = model._last_cpt_report
            before = report["new_pairs"]["cosine_before"]
            after = report["new_pairs"]["cosine_after"]
            deltas.extend((after - before).tolist())

    n_improved = sum(1 for d in deltas if d > 0)
    fraction_improved = n_improved / len(deltas)
    mean_delta = sum(deltas) / len(deltas)

    assert fraction_improved >= MIN_FRACTION_IMPROVED, (
        f"only {n_improved}/{len(deltas)} pairs improved ({fraction_improved:.3f}) "
        f"across {N_SEEDS} seeds, below the {MIN_FRACTION_IMPROVED} threshold")
    assert mean_delta >= MIN_MEAN_DELTA, (
        f"mean cosine delta {mean_delta:.4f} across {N_SEEDS} seeds is below "
        f"the {MIN_MEAN_DELTA} threshold")
