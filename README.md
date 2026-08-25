# ADCLIP
 Retrieve NRPS adenylation-domain ↔ substrate matches with a dual-encoder model, plus CPT to teach it new pairs.




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
adclip query_adomain --fasta new.fasta [--substrates smiles.csv] [--pool training|path.fasta] \
    [--checkpoint baseline|complete|path] [--top_k 20] [--out results.csv]

adclip query_substrate --smiles "..." [--corpus_fasta candidates.fasta] [--pool training|path.fasta] \
    [--checkpoint baseline|complete|path] [--top_k 20] [--out results.csv]

adclip cpt --pairs new_pairs.csv --checkpoint baseline \
    [--lr 1e-4] [--epochs 50] [--patience 5] [--l_atp 0.2] [--l_prop 0.3] [--batch_size 64] \
    [--pool training|path.fasta] [--val_pairs val_pairs.csv] [--val_split substrate|row] \
    --out my_checkpoint.pt
```

`--substrates smiles.csv` columns: `name,smiles`.
`--pairs new_pairs.csv` columns: `a_domain_sequence,smiles[,substrate_name]`.

## Python API

```python
from adclip import ADCLIP

model = ADCLIP.load(checkpoint="complete")  # or "baseline", or a .pt path

# A-domain -> ranked substrates
df = model.query_adomain(["MSTA...", "GILV..."])

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
  Epoch count (7) picked via 5-fold cross-validation.


## Repo layout

```
src/adclip/              the package
  data/                  checkpoints, fingerprints, 1AMU
                         reference, training corpus
tests/                  alignment round-trip, query, and CPT smoke tests
docs/                   environment prerequisites
```
