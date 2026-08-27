"""
Continual pretraining: fold new (A-domain sequence, activating substrate
SMILES) pairs into the existing training split and continue training from a
chosen checkpoint.

"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from . import alignment, config, corpus, features
from . import substrates as substrates_module
from .model import DualEncoder, PairDataset

DEFAULT_SEED = 61
DEFAULT_VAL_SIZE = 0.15
DEFAULT_BETA = 0.99



def compute_class_weights(df, beta=DEFAULT_BETA):
    counts = df["substrate"].value_counts()
    eff = {s: (1 - pow(beta, f)) / (1 - beta) for s, f in counts.items()}
    w = {s: 1.0 / e for s, e in eff.items()}
    total = sum(w.values())
    return {k: v / total * len(w) for k, v in w.items()}


def build_balanced_sampler(df, weights_dict, seed=None):
    sample_weights = [weights_dict[s] for s in df["substrate"]]
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                  replacement=True, generator=generator)


def split_train_val(corpus_df, seed, val_size, held_out_substrates):
    tv_df = corpus_df[~corpus_df["substrate"].isin(held_out_substrates)].copy().reset_index(drop=True)
    unique_substrates = tv_df["substrate"].unique()
    train_subs, val_subs = train_test_split(unique_substrates, test_size=val_size, random_state=seed)
    train_df = tv_df[tv_df["substrate"].isin(train_subs)].reset_index(drop=True)
    val_df = tv_df[tv_df["substrate"].isin(val_subs)].reset_index(drop=True)
    return train_df, val_df


def split_train_val_by_row(corpus_df, seed, val_size, held_out_substrates):
    tv_df = corpus_df[~corpus_df["substrate"].isin(held_out_substrates)].copy().reset_index(drop=True)
    train_parts, val_parts = [], []
    for _, group in tv_df.groupby("substrate"):
        if len(group) < 2:
            train_parts.append(group)
            continue
        train_g, val_g = train_test_split(group, test_size=val_size, random_state=seed)
        train_parts.append(train_g)
        val_parts.append(val_g)
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else tv_df.iloc[0:0]
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else tv_df.iloc[0:0]
    return train_df, val_df


def _substrate_smiles_map(corpus_df):
    dedup = corpus_df.drop_duplicates("substrate")
    return dict(zip(dedup["substrate"], dedup["smiles"]))


def _resolve_new_pairs(pairs_df, known_substrate_names, aligned_info):
    known_lookup = {n.strip().lower(): n for n in known_substrate_names}
    rows = []
    for row in pairs_df.itertuples():
        raw_name = getattr(row, "substrate_name", None)
        norm = raw_name.strip().lower() if isinstance(raw_name, str) and raw_name.strip() else None
        if norm is not None and norm in known_lookup:
            substrate_key = known_lookup[norm]
            display_name = substrate_key
            own_smiles = None 
        else:
            canon = features.canonical_smiles(row.smiles)
            substrate_key = canon
            display_name = raw_name if norm else canon
            own_smiles = row.smiles
        info = aligned_info[row.new_domain_id]
        rows.append({
            "new_domain_id": row.new_domain_id,
            "substrate": substrate_key,
            "display_name": display_name,
            "own_smiles": own_smiles,
            "a_domain_sequence": row.a_domain_sequence,
            "code_idx": info["code_idx"],
            "unresolved_positions": info["unresolved_positions"],
        })
    return pd.DataFrame(rows)


def _align_and_resolve_pairs(pairs_df, known_substrate_names, pool, threads, id_prefix):
    pairs_df = pairs_df.reset_index(drop=True).copy()
    pairs_df["new_domain_id"] = [f"{id_prefix}_{i}" for i in range(len(pairs_df))]
    new_seqs = dict(zip(pairs_df["new_domain_id"], pairs_df["a_domain_sequence"]))
    aligned_info = alignment.align_new_sequences(new_seqs, pool=pool, threads=threads)
    return _resolve_new_pairs(pairs_df, known_substrate_names, aligned_info)


def _compute_desc_stats(substrate_names, corpus_df, new_pair_smiles: dict):
    known_names, _ = substrates_module.load_fingerprints()
    known_set = set(known_names)
    sub_smiles_map = _substrate_smiles_map(corpus_df)
    raw = [
        features.compute_descriptors(sub_smiles_map[sub] if sub in known_set else new_pair_smiles[sub])
        for sub in substrate_names
    ]
    raw = np.array(raw, dtype=np.float64)
    return raw.mean(axis=0).tolist(), raw.std(axis=0).tolist()


def _build_substrate_pool(substrate_names, corpus_df, desc_mean, desc_std, new_pair_smiles: dict):
    known_names, known_fps = substrates_module.load_fingerprints()
    known_fp_lookup = dict(zip(known_names, known_fps))
    sub_smiles_map = _substrate_smiles_map(corpus_df)

    aa_rows, desc_map, aa_index_map = [], {}, {}
    for i, sub in enumerate(substrate_names):
        aa_index_map[sub] = i
        if sub in known_fp_lookup:
            aa_rows.append(known_fp_lookup[sub])
            smi = sub_smiles_map[sub]
        else:
            smi = new_pair_smiles[sub]
            aa_rows.append(features.compute_morgan_fingerprint(smi))
        raw_desc = features.compute_descriptors(smi)
        desc_map[sub] = features.normalize_descriptors(raw_desc, desc_mean, desc_std)
    return np.stack(aa_rows).astype(np.float32), aa_index_map, desc_map


def _catastrophic_forgetting_check(model, device, train_df, aa_emb, aa_index_map,
                                    k_values=(50, 100, 200), max_substrates=10):
    ad_emb = corpus._onehot_matrix(train_df)
    latents = model.project_adomain_latent(ad_emb, device=device)
    index = corpus.FaissCorpusIndex(train_df["new_domain_id"].tolist(), latents)
    true_sub_by_id = dict(zip(train_df["new_domain_id"], train_df["substrate"]))

    substrates = sorted(train_df["substrate"].unique())[:max_substrates]
    rows = []
    for sub in substrates:
        q = model.project_aa_latent(
            torch.tensor(aa_emb[aa_index_map[sub]], dtype=torch.float32).to(device)
        ).detach().cpu().numpy()
        hits = index.search(q, top_k=max(k_values))
        y_true = np.array([1 if true_sub_by_id[hid] == sub else 0 for hid, _ in hits])
        y_score = np.array([sim for _, sim in hits])
        n_relevant = int((train_df["substrate"] == sub).sum())
        row = {"substrate": sub}
        for k in k_values:
            row[f"recall@{k}"] = y_true[:k].sum() / max(1, n_relevant)
            row[f"ndcg@{k}"] = (
                ndcg_score(y_true[:k].reshape(1, -1), y_score[:k].reshape(1, -1))
                if y_true[:k].sum() > 0 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def continue_pretrain(model: DualEncoder, base_checkpoint_name: str, pairs_df: pd.DataFrame,
                        device, lr=1e-4, epochs=50, patience=5, l_atp=0.2, l_prop=0.3,
                        batch_size=64, pool="training", threads=4,
                        seed=DEFAULT_SEED, val_size=DEFAULT_VAL_SIZE, beta=DEFAULT_BETA,
                        val_pairs_df: pd.DataFrame = None, val_split: str = "substrate"):

    torch.manual_seed(seed)

    meta = config.checkpoint_meta(base_checkpoint_name)
    held_out = meta["held_out_substrates"]
    desc_mean, desc_std = meta["desc_mean"], meta["desc_std"]

    print(f"[cpt] base checkpoint '{base_checkpoint_name}': "
          f"held_out_substrates={held_out or '(none)'}")

    corpus_df = corpus.load_default_corpus_df()
    known_names, _ = substrates_module.load_fingerprints()


    print(f"[cpt] aligning {len(pairs_df)} new A-domain sequence(s)...")
    new_pairs_df = _align_and_resolve_pairs(pairs_df, known_names, pool, threads, "cpt_pair")
    print("[cpt] new pairs resolved to substrates:")
    for row in new_pairs_df.itertuples():
        tag = "existing" if row.own_smiles is None else "brand-new"
        print(f"    {row.new_domain_id} -> {row.display_name} ({tag})"
              + (f" [unresolved anchors: {row.unresolved_positions}]" if row.unresolved_positions else ""))

   
    val_pair_smiles = {}
    if val_pairs_df is not None:

        train_df = corpus_df[~corpus_df["substrate"].isin(held_out)].copy().reset_index(drop=True)
        val_resolved_df = _align_and_resolve_pairs(val_pairs_df, known_names, pool, threads, "cpt_val_pair")
        val_df = val_resolved_df[["new_domain_id", "substrate", "a_domain_sequence", "code_idx"]]
        val_pair_smiles = dict(zip(val_resolved_df["substrate"], val_resolved_df["own_smiles"]))
    else:
        if not held_out:
            raise ValueError(
                f"Base checkpoint '{base_checkpoint_name}' has no held-out substrates (it was "
                f"trained on the full corpus) -- there's no data left to "
                f"validate against automatically. Pass val_pairs_df (CLI: --val_pairs) with "
                f"your own validation examples."
            )
        if val_split == "substrate":
            train_df, val_df = split_train_val(corpus_df, seed, val_size, held_out)
        elif val_split == "row":
            train_df, val_df = split_train_val_by_row(corpus_df, seed, val_size, held_out)
        else:
            raise ValueError(f"val_split must be 'substrate' or 'row', got {val_split!r}")

    print(f"[cpt] train/val split: {len(train_df)} train rows "
          f"({train_df['substrate'].nunique()} substrates), "
          f"{len(val_df)} val rows ({val_df['substrate'].nunique()} substrates)")

  
    new_pair_smiles = dict(zip(new_pairs_df["substrate"], new_pairs_df["own_smiles"]))
    train_substrates = sorted(set(train_df["substrate"]) | set(new_pairs_df["substrate"]))

    desc_mean, desc_std = _compute_desc_stats(train_substrates, corpus_df, new_pair_smiles)
    print(f"[cpt] recomputed descriptor normalization stats over {len(train_substrates)} substrates "
          f"(was frozen from '{base_checkpoint_name}' before)")

    cp_aa_emb, cp_aa_index_map, cp_desc_map = _build_substrate_pool(
        train_substrates, corpus_df, desc_mean, desc_std, new_pair_smiles)

    cp_df = pd.concat([
        train_df[["new_domain_id", "substrate", "a_domain_sequence", "code_idx"]],
        new_pairs_df[["new_domain_id", "substrate", "a_domain_sequence", "code_idx"]],
    ], ignore_index=True)
    cp_ad_emb = corpus._onehot_matrix(cp_df)

    class_weights = compute_class_weights(cp_df, beta=beta)
    sampler = build_balanced_sampler(cp_df, class_weights, seed=seed)
    cp_loader = DataLoader(
        PairDataset(cp_aa_emb, cp_ad_emb, cp_df, cp_aa_index_map, cp_desc_map),
        batch_size=batch_size, sampler=sampler)

   
    val_substrates = sorted(val_df["substrate"].unique())
    val_aa_emb, val_aa_index_map, val_desc_map = _build_substrate_pool(
        val_substrates, corpus_df, desc_mean, desc_std, val_pair_smiles)
    val_ad_emb = corpus._onehot_matrix(val_df)
    val_loader = DataLoader(
        PairDataset(val_aa_emb, val_ad_emb, val_df, val_aa_index_map, val_desc_map),
        batch_size=batch_size, shuffle=False)

    held_out_df = corpus_df[corpus_df["substrate"].isin(held_out)].copy().reset_index(drop=True)
    if len(held_out_df):
        ho_aa_emb, ho_aa_index_map, _ = _build_substrate_pool(
            sorted(held_out_df["substrate"].unique()), corpus_df, desc_mean, desc_std, {})

    model.eval()
    before_rows = []
    with torch.no_grad():
        for row in new_pairs_df.itertuples():
            z_aa = model.project_aa_latent(
                torch.tensor(cp_aa_emb[cp_aa_index_map[row.substrate]], dtype=torch.float32).to(device))
            z_ad = model.project_adomain_latent(
                features.build_adomain_onehot(row.a_domain_sequence, row.code_idx)[None], device=device)[0]
            cosine = float(z_aa.detach().cpu().numpy() @ z_ad)
            before_rows.append({
                "new_domain_id": row.new_domain_id, "substrate": row.display_name,
                "cosine_before": cosine,
                "code": features.code_idx_to_residues(row.a_domain_sequence, row.code_idx),
                "code_idx": row.code_idx,
            })
    forgetting_before = _catastrophic_forgetting_check(model, device, train_df, cp_aa_emb, cp_aa_index_map)
    zero_shot_before = (
        _catastrophic_forgetting_check(model, device, held_out_df, ho_aa_emb, ho_aa_index_map)
        if len(held_out_df) else None
    )


    model.optimize(cp_loader, val_loader, weights_dict=class_weights, epochs=epochs,
                    patience=patience, l_atp=l_atp, l_prop=l_prop, lr=lr, device=device)

   
    model.eval()
    after_by_id = {}
    with torch.no_grad():
        for row in new_pairs_df.itertuples():
            z_aa = model.project_aa_latent(
                torch.tensor(cp_aa_emb[cp_aa_index_map[row.substrate]], dtype=torch.float32).to(device))
            z_ad = model.project_adomain_latent(
                features.build_adomain_onehot(row.a_domain_sequence, row.code_idx)[None], device=device)[0]
            after_by_id[row.new_domain_id] = float(z_aa.detach().cpu().numpy() @ z_ad)
    forgetting_after = _catastrophic_forgetting_check(model, device, train_df, cp_aa_emb, cp_aa_index_map)
    zero_shot_after = (
        _catastrophic_forgetting_check(model, device, held_out_df, ho_aa_emb, ho_aa_index_map)
        if len(held_out_df) else None
    )

    new_pairs_report = pd.DataFrame(before_rows)
    new_pairs_report["cosine_after"] = new_pairs_report["new_domain_id"].map(after_by_id)

    forgetting_report = forgetting_before.merge(
        forgetting_after, on="substrate", suffixes=("_before", "_after"))

    updated_held_out = sorted(set(held_out) - set(new_pairs_df["substrate"]))
    if set(updated_held_out) != set(held_out):
        print(f"[cpt] held_out_substrates shrinks to {updated_held_out or '(none)'} "
              f"for the saved checkpoint (this round trained on some of them)")

    report = {"new_pairs": new_pairs_report, "forgetting_check": forgetting_report,
              "desc_mean": desc_mean, "desc_std": desc_std, "held_out_substrates": updated_held_out}
    if zero_shot_before is not None:
        report["zero_shot_check"] = zero_shot_before.merge(
            zero_shot_after, on="substrate", suffixes=("_before", "_after"))

    return model, report
