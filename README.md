# adclip

![ADCLIP architecture](docs/ADCLIP.pdf)

ADCLIP frames A-domain specificity as a dual-modality retrieval problem. A substrate encoder and an A-domain encoder jointly map their respective inputs into a shared latent space, where compatibility is measured by cosine similarity. The model is trained contrastively, pulling matched substrate--A-domain pairs together and pushing mismatched pairs apart. At inference, retrieval is bidirectional: given a query substrate, ADCLIP ranks A-domains by similarity, and vice versa.


- **`query_adomain`** — give one or more raw A-domain sequences, get every
  known substrate ranked by cosine similarity.
- **`query_substrate`** — give a substrate SMILES, get a corpus of A-domain
  sequences ranked by cosine similarity (the bundled training corpus by
  default, or your own FASTA).
- **`cpt`** — continual pretraining: give new (A-domain sequence, activating
  substrate SMILES) pairs, they get folded into the existing training data,
  and training continues from a chosen checkpoint.

## Install

```
pip install -e .
```

Also requires the **MUSCLE** alignment binary on `PATH` — not a pip
package. See `docs/ENVIRONMENT.md`.

## Why alignment at all?

The model is based on a fixed set of 16 specificity-determining residues, 
numbered against the reference sequence (PDB: 1AMU, https://www.rcsb.org/structure/1AMU).
A raw sequence has no inherent "position 210" until it's aligned against that reference.
So every new sequence goes through a quick MUSCLE alignment step first

## CLI

```
adclip query_adomain --fasta new.fasta [--substrates smiles.csv] [--alignment_context training|path.fasta] \
    [--checkpoint baseline|complete|path] [--top_k 20] [--out results.csv]

adclip query_substrate --smiles "..." [--corpus_fasta candidates.fasta] [--pool training|path.fasta] \
    [--checkpoint baseline|complete|path] [--top_k 20] [--out results.csv]

adclip cpt --pairs new_pairs.csv --checkpoint baseline \
    [--lr 1e-4] [--epochs 50] [--patience 5] [--l_atp 0.2] [--l_prop 0.3] [--batch_size 64] \
    [--pool training|path.fasta] [--threads 4] [--val_pairs val_pairs.csv] [--val_split substrate|row] \
    --out my_checkpoint.pt
```

`--substrates smiles.csv` columns: `name,smiles`.
`--pairs new_pairs.csv` columns: `a_domain_sequence`, plus `smiles` (for a
brand-new substrate) and/or `substrate_name` (for a substrate the checkpoint
already knows).

## Python API

```python
from adclip import ADCLIP

model = ADCLIP.load(checkpoint="complete")  # or "baseline", or a .pt path

# A-domain -> ranked substrates
df = model.query_adomain({"domain_1": "MSTA...", "domain_2": "GILV..."})

# substrate -> ranked A-domains (SMILES required)
df = model.query_substrate("C1CCNC(C1)C(=O)O")

# continual pretraining on new pairs
model.continue_pretrain("new_pairs.csv", out_checkpoint="my_checkpoint.pt")
```

## Two checkpoints

- **`baseline`** — zero-shot: hasn't seen 5 held-out test substrates
  (`p-hydroxyphenylglycine`, `piperazic_acid`, `p-aminobenzoic_acid`,
  `2-aminoadipic_acid`, `pipecolic_acid`).
- **`complete`** — trained on the full dataset, all 43 substrates included.
  Epoch count (9) picked via cross-validation.

### Paper checkpoints

`data/checkpoints/paper_checkpoints/` holds the reference checkpoints behind
the paper's few-shot claim, organized by training step:

```
paper_checkpoints/
  zeroshot/            --- baseline_zeroshot.pt (0 shots — same model as `baseline`)
  step1/step1_plan{0..4}.pt   --- after 1 shot of each held-out substrate, 5 independent plans
  step2/step2_plan{0..4}.pt   --- after 2 shots
  step3/step3_plan{0..4}.pt   --- after 3 shots
```

Each `plan` is an independent run with its own random ordering of which
examples get added at each step. These aren't registered as named
checkpoints (`config.checkpoint_path`/`ADCLIP.load` don't know about them) —
load one directly by its file path instead.


## Repo layout

```
src/adclip/      --- the package
  data/          --- checkpoints, fingerprints, 1AMU reference, training corpus
    checkpoints/
      complete.pt
      paper_checkpoints/
        zeroshot/baseline_zeroshot.pt
        step1/, step2/, step3/  --- reference checkpoints for the paper's few-shot claim
tests/           --- alignment round-trip, query, and CPT smoke tests
docs/            --- environment prerequisites
```
