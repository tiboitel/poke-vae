import json
import torch
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

            for (x,) in self.loader:
                x = x.to(self.device)

                self.optimizer.zero_grad()
                loss = vae_loss(self.model(x), epoch, self.cfg)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss {total_loss:.2f}")

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
