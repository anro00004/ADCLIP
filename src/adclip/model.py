
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class PairDataset(Dataset):

    def __init__(self, aa_emb, ad_emb, df, aa_index_map, desc_map):
        self.aa_emb = aa_emb
        self.ad_emb = ad_emb
        self.df = df
        self.aa_index_map = aa_index_map
        self.desc_map = desc_map

    def __len__(self):
        return len(self.ad_emb)

    def __getitem__(self, idx):
        substrate = self.df.iloc[idx]["substrate"]
        prop_target = torch.tensor(self.desc_map[substrate], dtype=torch.float32)
        return (
            torch.tensor(self.aa_emb[self.aa_index_map[substrate]], dtype=torch.float32),
            torch.tensor(self.ad_emb[idx], dtype=torch.float32),
            substrate,
            prop_target,
        )


class DualEncoder(nn.Module):

    def __init__(self, amino_acid_dim, adomain_dims, latent_dim, dropout,
                 init_temperature, n_descriptors=12):
        super().__init__()
        self.latent_dim = latent_dim
        self.adomain_dims = adomain_dims

        self.aa_encoder = nn.Sequential(
            nn.Linear(amino_acid_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.prop_head = nn.Linear(latent_dim, n_descriptors)

        self.adomain_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(adomain_dims[i], 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, latent_dim),
            )
            for i in range(len(adomain_dims))
        ])

        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature)))
        

   
    def _encode_adomain(self, ad_emb):
        node_h = torch.stack([
            self.adomain_encoders[i](ad_emb[:, i, :self.adomain_dims[i]])
            for i in range(ad_emb.size(1))
        ], dim=1)
        B, L, _ = node_h.shape
        weights = torch.full((B, L), 1.0 / L, device=node_h.device, dtype=node_h.dtype)
        z_ad = F.normalize(node_h.mean(dim=1), dim=-1)
        return z_ad, weights


    def forward(self, aa_emb, ad_emb):
        z_aa = F.normalize(self.aa_encoder(aa_emb), dim=-1)
        prop_pred = self.prop_head(z_aa)
        z_ad, weights = self._encode_adomain(ad_emb)
        return z_aa, z_ad, torch.exp(self.log_temperature), weights, prop_pred

   
    def project_aa_latent(self, aa_emb):
        if aa_emb.dim() == 1:
            aa_emb = aa_emb.unsqueeze(0)
        result = F.normalize(self.aa_encoder(aa_emb), dim=-1)
        return result.squeeze(0) if result.shape[0] == 1 else result

    def project_adomain_latent(self, ad_emb, device, batch_size=64):
        if isinstance(ad_emb, list):
            ad_emb = np.stack(ad_emb, axis=0)
        all_latent = []
        for i in range(0, len(ad_emb), batch_size):
            batch_emb = torch.tensor(ad_emb[i:i + batch_size], dtype=torch.float32).to(device)
            z_ad, _ = self._encode_adomain(batch_emb)
            all_latent.append(z_ad.detach().cpu().numpy().astype("float32"))
        return np.vstack(all_latent)

    
    def compute_class_balanced_loss(self, z_aa, z_ad, temperature, sub, weights_dict):
        logits = z_aa @ z_ad.T / temperature
        sub_array = np.array(sub)
        fn_mask = torch.tensor(
            sub_array[:, None] == sub_array[None, :],
            dtype=torch.bool, device=logits.device)
        fn_mask.fill_diagonal_(False)
        logits = logits.masked_fill(fn_mask, float("-inf"))
        labels = torch.arange(z_aa.size(0), device=logits.device)
        w = torch.tensor([weights_dict[s] for s in sub], dtype=torch.float32, device=logits.device)
        crit = nn.CrossEntropyLoss(reduction="none")
        return ((crit(logits, labels) * w + crit(logits.T, labels) * w) / 2).mean()

    def compute_loss(self, z_aa, z_ad, temperature, sub):
        logits = z_aa @ z_ad.T / temperature
        sub_array = np.array(sub)
        fn_mask = torch.tensor(
            sub_array[:, None] == sub_array[None, :],
            dtype=torch.bool, device=logits.device)
        fn_mask.fill_diagonal_(False)
        logits = logits.masked_fill(fn_mask, float("-inf"))
        labels = torch.arange(z_aa.size(0), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def compute_atp_loss(self, z_aa, z_ad):
        return sum(
            torch.linalg.vector_norm(z_ad[i] - z_aa[i]) ** 2 / z_ad.shape[0]
            for i in range(len(z_ad)))

  
    def optimize(self, train_dataloader, val_dataloader, weights_dict,
                 epochs, patience, l_atp, l_prop, lr, device, weight_decay=1e-3):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for aa_batch, ad_batch, substrate, prop_targets in train_dataloader:
                aa_batch = aa_batch.to(device)
                ad_batch = ad_batch.to(device)
                prop_targets = prop_targets.to(device)

                z_aa, z_ad, temperature, _, prop_pred = self.forward(aa_batch, ad_batch)

                prop_loss = F.mse_loss(prop_pred, prop_targets)
                atp_loss = self.compute_atp_loss(z_aa, z_ad)
                contrastive_loss = self.compute_class_balanced_loss(
                    z_aa, z_ad, temperature, substrate, weights_dict)

                loss = contrastive_loss + l_atp * atp_loss + l_prop * prop_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            total_loss /= len(train_dataloader)
            train_losses.append(total_loss)

            if val_dataloader is None:
                scheduler.step(total_loss)
                print(f"Epoch {epoch + 1:02d}/{epochs} | Train: {total_loss:.4f} | "
                      f"(no val — full-data mode) | tau: {torch.exp(self.log_temperature).item():.4f}")
                continue

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for aa_val, ad_val, substrate_val, _ in val_dataloader:
                    aa_val = aa_val.to(device)
                    ad_val = ad_val.to(device)
                    z_aa_v, z_ad_v, temp_v, _, _ = self.forward(aa_val, ad_val)
                    contrastive_loss = self.compute_loss(z_aa_v, z_ad_v, temp_v, substrate_val)
                    atp_loss = self.compute_atp_loss(z_aa_v, z_ad_v)
                    val_loss += (contrastive_loss + l_atp * atp_loss).item()

            val_loss /= len(val_dataloader)
            scheduler.step(val_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1:02d}/{epochs} | Train: {total_loss:.4f} | "
                  f"Val: {val_loss:.4f} | tau: {torch.exp(self.log_temperature).item():.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"Restored best model (val_loss={best_val_loss:.4f})")

        return train_losses, val_losses
