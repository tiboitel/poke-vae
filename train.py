import json
import torch
import math
from torch.utils.data import DataLoader

from model import VAE
from loss import vae_loss
from data import load_dataset

class PokeVAETrainer:
    """
    Minimal training API for PokeVAE.
    Owns model, optimizer, data, and checkpointing.
    """

    def __init__(self, cfg: dict, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ---- Data ----
        dataset, scaler, type_to_idx, talent_to_idx, _ = load_dataset(cfg)
        self.loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
        )

        self.scaler = scaler
        self.type_to_idx = type_to_idx
        self.talent_to_idx = talent_to_idx

        # ---- Model ----
        self.model = VAE(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["learning_rate"],
        )

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device | None = None):
        """
        Resume training from a saved checkpoint.
        """
        ckpt = torch.load(path, map_location="cpu")
        trainer = cls(ckpt["config"], device=device)

        trainer.model.load_state_dict(ckpt["model"])

        # Restore scaler
        trainer.scaler.mean_ = ckpt["scaler_mean"].numpy()
        trainer.scaler.scale_ = ckpt["scaler_scale"].numpy()
        trainer.scaler.n_features_in_ = trainer.scaler.mean_.shape[0]

        trainer.type_to_idx = ckpt["type_to_idx"]
        trainer.talent_to_idx = ckpt["talent_to_idx"]

        return trainer
    
    def train(self):
        print("--- Start training PokeVAE. ---")
        for epoch in range(self.cfg["epochs"]):
            total_loss = 0.0
            total_kl = 0.0
            kl_accumulator = []

            for (x,) in self.loader:
                x = x.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                out = self.model(x)  # out has 8 elements

                # Compute loss
                metrics = vae_loss(out, epoch, self.cfg)
                loss = metrics["loss"]

                # Extract mu and logvar for monitoring
                mu, logvar = out[6], out[7]
                avg_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean().item()
                total_kl += avg_kl
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                kl_accumulator.append(metrics["kl_raw_per_dim"].cpu())

            if epoch % 100 == 0:
                with torch.no_grad():
                    kl_per_dim = torch.stack(kl_accumulator).mean(dim=0)

                    kl_mean = kl_per_dim.mean().item()
                    kl_std = kl_per_dim.std(unbiased=False).item()

                    latent_dim = kl_per_dim.numel()
                    active_dims = (kl_per_dim > 0.01).sum().item()

                    recon_loss = (
                        metrics["stats_loss"]
                        + metrics["type_loss"]
                        + metrics["talent_loss"]
                    )

                    # ---- Integrity score ----
                    collapse_term = active_dims / latent_dim

                    target_kl_ratio = 0.7
                    kl_balance_term = math.exp(
                        -abs((kl_std / (kl_mean + 1e-8)) - target_kl_ratio)
                    )

                    kl_target = self.cfg["latent_dim"] * self.cfg["kl_free_bits"]
                    capacity_term = min(1.0, kl_mean / kl_target)

                    recon_baseline = 5.0  # dataset dependent, tune once
                    recon_term = math.exp(-recon_loss / recon_baseline)

                    score = (
                        collapse_term
                        * kl_balance_term
                        * capacity_term
                        * recon_term
                    )

                    warnings = []
                    if collapse_term < 0.7:
                        warnings.append("posterior_collapse")
                    if kl_std / (kl_mean + 1e-8) > 2.0:
                        warnings.append("latent_dominance")
                    if kl_balance_term < 0.5:
                        warnings.append("kl_imbalance")

                    print(
                        f"Epoch {epoch:04d} | "
                        f"Loss {total_loss:.2f} | "
                        f"Stats loss {metrics['stats_loss']:.2f} | "
                        f"Type loss {metrics['type_loss']:.2f} | "
                        f"Ability loss {metrics['talent_loss']:.2f} | "
                        f"Active {active_dims}/{latent_dim} | "
                        f"KL {kl_mean:.4f} ± {kl_std:.4f} | "
                        f"Score {score:.3f}"
                    )

                    if warnings:
                        print(f"Warnings: {', '.join(warnings)}")


    def save(self, path: str | None = None):
        """
        Save a full training artifact (model + preprocessing + config).
        """
        path = path or self.cfg["checkpoint_path"]

        torch.save(
            {
                "model": self.model.state_dict(),
                "scaler_mean": torch.tensor(
                    self.scaler.mean_, dtype=torch.float32
                ),
                "scaler_scale": torch.tensor(
                    self.scaler.scale_, dtype=torch.float32
                ),
                "type_to_idx": self.type_to_idx,
                "talent_to_idx": self.talent_to_idx,
                "config": self.cfg,
            },
            path,
        )

def main():
    cfg = json.load(open("config.json"))

    trainer = PokeVAETrainer(cfg)
    trainer.train()
    trainer.save()

if __name__ == "__main__":
    main()
