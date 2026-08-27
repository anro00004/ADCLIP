# Environment

## Python dependencies

```
pip install -e .
```

pulls in `torch`, `pandas`, `numpy`, `faiss-cpu`, `rdkit`, `biopython`,
`scikit-learn` — `pyproject.toml` 

**Exact versions the current checkpoints**

| package   | version |
|-----------|---------|
| Python    | 3.12.12 |
| torch     | 2.9.0+cu128 |
| pandas    | 2.3.3 |
| numpy     | 2.4.6 |
| faiss     | 1.12.0 |
| rdkit     | 2025.09.1 |
| biopython | 1.85 |
| scikit-learn | 1.8.0 |
| muscle       | 5.3.linux64 |



## MUSCLE (required, not a pip package)

Alignment (`adclip.alignment`) shells out to the `muscle` binary. 
This is how new A-domain sequences get placed onto the 1AMU reference frame to recover their anchor-position (proposed by https://www.biorxiv.org/content/10.64898/2026.06.15.732251v1). 
It is **not** installable via pip.

Pinned version for reproducibility: **MUSCLE 5.3**
(https://github.com/rcedgar/muscle).

Install via conda:

```
conda install -c conda-forge muscle
```
Verify with:

```
muscle -version
```

`alignment.py` resolves the binary via `shutil.which("muscle")` at call
time and raises a clear error (pointing back to this doc) if it isn't on
`PATH`.


## GPU

Not required. `AdomainSubstrateModel.load(device=...)` and
`continual_pretrain(...)` accept `"cpu"` or `"cuda"`.
The model is small (64-d latent space), 
CPU is fine for query and 
for CPT on a handful of new pairs.
