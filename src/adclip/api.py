
import json
import numpy as np
import pandas as pd
import torch
import os
from . import alignment, config, corpus, cpt as cpt_module, features, retrieval
from . import substrates as substrates_module
from .model import DualEncoder


class ADCLIP:
    def __init__(self, model: DualEncoder, device: str, checkpoint_name: str):
        self.model = model
        self.device = device
        self.checkpoint_name = checkpoint_name  # "paper checkpoint" | "complete" | a path


    @classmethod
    def load(cls, checkpoint: str = "complete", device: str = "cpu") -> "ADCLIP":
        path = config.checkpoint_path(checkpoint)
        model = DualEncoder(**config.MODEL_CTOR_ARGS)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return cls(model, device, checkpoint)

    def save(self, out_path, meta: dict = None):
        torch.save(self.model.state_dict(), out_path)
        print(f"Saved checkpoint: {out_path}")
        if meta is not None:
            sidecar = config.sidecar_meta_path(out_path)
            with open(sidecar, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Saved checkpoint metadata: {sidecar}")

  
    def query_adomain(self, sequences, substrate_smiles: dict = None,
                       pool: str = "training", top_k: int = None,
                       threads: int = 4) -> pd.DataFrame:
        """

        sequences: {id: raw_sequence} dict, or a FASTA path.
        substrate_smiles: optional
        {name: smiles} pool to rank against; None -> the 43-substrate default set of corpus.

        """
        if isinstance(sequences, str):
            if not os.path.exists(sequences):
                raise FileNotFoundError(f"FASTA path not found: {sequences!r}")
            seq_dict = alignment.read_fasta(sequences)
        else:
            seq_dict = dict(sequences)

        aligned = alignment.align_new_sequences(seq_dict, pool=pool, threads=threads)

        if substrate_smiles:
            substrate_ids, substrate_latents = substrates_module.custom_substrates_latents(
                self.model, substrate_smiles, self.device)
        else:
            substrate_ids, substrate_latents = substrates_module.default_substrates_latents(self.model, self.device)

        frames = []
        for seq_id, seq in seq_dict.items():
            code_idx = aligned[seq_id]["code_idx"]
            onehot = features.build_adomain_onehot(seq, code_idx)[None]
            z_ad = self.model.project_adomain_latent(onehot, device=self.device)[0]
            frames.append(retrieval.rank_against_pool(
                seq_id, z_ad, substrate_ids, substrate_latents, target_col="substrate", top_k=top_k,
                unresolved_positions=aligned[seq_id]["unresolved_positions"],
                code=features.code_idx_to_residues(seq, code_idx), code_idx=code_idx))
        return pd.concat(frames, ignore_index=True)

   
    def query_substrate(self, smiles: str, corpus_fasta: str = None,
                            pool: str = "training", top_k: int = None,
                            threads: int = 4) -> pd.DataFrame:

        z_aa = substrates_module.generate_latent_from_smiles(self.model, smiles, self.device)

        if corpus_fasta is None:
            corpus_df = corpus.load_default_corpus_df()
            adomain_ids, adomain_latents, code_map, code_idx_map = corpus.build_latent_adomains_from_df(
                self.model, corpus_df, self.device)
            unresolved_map = {i: [] for i in adomain_ids}
        else:
            adomain_ids, adomain_latents, unresolved_map, code_map, code_idx_map = corpus.build_latent_adomains_from_fasta(
                self.model, corpus_fasta, self.device, pool=pool, threads=threads)

        df = retrieval.rank_against_pool(smiles, z_aa, adomain_ids, adomain_latents,
                                          target_col="a_domain_id", top_k=top_k)
        df["unresolved_positions"] = df["a_domain_id"].map(unresolved_map)
        df["code"] = df["a_domain_id"].map(code_map)
        df["code_idx"] = df["a_domain_id"].map(code_idx_map)
        return df


    def continue_pretrain(self, pairs_csv: str, lr: float = 1e-4, epochs: int = 50,
                            patience: int = 5, l_atp: float = 0.2, l_prop: float = 0.3,
                            batch_size: int = 64, pool: str = "training", threads: int = 4,
                            val_pairs_csv: str = None, val_split: str = "substrate",
                            out_checkpoint: str = None) -> "ADCLIP":
        def _validate_pairs_columns(df, label):
            if "a_domain_sequence" not in df.columns:
                raise ValueError(f"{label} csv is missing required column: 'a_domain_sequence'")
            if "smiles" not in df.columns and "substrate_name" not in df.columns:
                raise ValueError(
                    f"{label} csv must have a 'smiles' column (for brand-new substrates) "
                    f"and/or a 'substrate_name' column (for substrates the checkpoint already knows)")

        pairs_df = pd.read_csv(pairs_csv)
        _validate_pairs_columns(pairs_df, "pairs")

        val_pairs_df = None
        if val_pairs_csv is not None:
            val_pairs_df = pd.read_csv(val_pairs_csv)
            _validate_pairs_columns(val_pairs_df, "val_pairs")

        base_meta = config.checkpoint_meta(self.checkpoint_name)

        self.model, report = cpt_module.continue_pretrain(
            self.model, self.checkpoint_name, pairs_df, self.device,
            lr=lr, epochs=epochs, patience=patience, l_atp=l_atp, l_prop=l_prop,
            batch_size=batch_size, pool=pool, threads=threads,
            val_pairs_df=val_pairs_df, val_split=val_split)

        print("\n[cpt] new-pair cosine, before vs after:")
        print(report["new_pairs"].to_string(index=False))
        print("\n[cpt] catastrophic-forgetting check (capped sample of train substrates), before vs after:")
        print(report["forgetting_check"].to_string(index=False))
        if "zero_shot_check" in report:
            print("\n[cpt] zero-shot check (substrates the base checkpoint never trained on), before vs after:")
            print(report["zero_shot_check"].to_string(index=False))

        if out_checkpoint:
            new_meta = {**base_meta, "desc_mean": report["desc_mean"], "desc_std": report["desc_std"],
                        "held_out_substrates": report["held_out_substrates"]}
            self.save(out_checkpoint, meta=new_meta)
        self.checkpoint_name = out_checkpoint or f"{self.checkpoint_name}+cpt"
        self._last_cpt_report = report
        return self
