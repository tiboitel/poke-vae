import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

from model import VAE
from data import encode_types, encode_talent
from utils import print_pokemon_entries


class PokeVAEGenerator:
    """
    Inference-only wrapper around a trained PokeVAE.
    Owns model, scaler, config, and vocabularies.
    """

    def __init__(
            self,
            model: VAE,
            cfg: dict,
            scaler: StandardScaler,
            type_to_idx: dict,
            talent_to_idx: dict,
            device: torch.device,
            ):
        self.model = model
        self.cfg = cfg
        self.scaler = scaler
        self.type_to_idx = type_to_idx
        self.talent_to_idx = talent_to_idx
        self.idx_to_type = {i: t for t, i in type_to_idx.items()}
        self.idx_to_talent = {i: t for t, i in talent_to_idx.items()}
        self.device = device

        self.model.eval()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device | None = None):
        ckpt = torch.load(path, map_location="cpu")

        cfg = ckpt["config"]
        device = device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
                )

        model = VAE(cfg).to(device)
        model.load_state_dict(ckpt["model"])

        scaler = StandardScaler()
        scaler.mean_ = ckpt["scaler_mean"].numpy()
        scaler.scale_ = ckpt["scaler_scale"].numpy()
        scaler.n_features_in_ = scaler.mean_.shape[0]

        return cls(
                model=model,
                cfg=cfg,
                scaler=scaler,
                type_to_idx=ckpt["type_to_idx"],
                talent_to_idx=ckpt["talent_to_idx"],
                device=device,
                )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_pokemon(self, pokemon):
        stats = np.array(
                [[pokemon["base"][s] for s in self.cfg["stats_order"]]],
                dtype=np.float32,
                )
        stats = self.scaler.transform(stats)

        types = encode_types(pokemon, self.type_to_idx)[None, :]
        talent_idx = encode_talent(pokemon, self.talent_to_idx)

        talents = np.zeros((1, self.cfg["talent_dim"]), dtype=np.float32)
        talents[0, talent_idx] = 1.0

        x = np.concatenate([stats, types, talents], axis=1)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            mu, logvar = self.model.encode(x)
            z = self.model.reparameterize(mu, logvar)

        return z

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_random(self, n: int = 1):
        z = torch.randn(n, self.cfg["latent_dim"], device=self.device)

        with torch.no_grad():
            stats, types, talents = self.model.decode(z)

        stats = self.scaler.inverse_transform(stats.cpu().numpy())

        types = (torch.sigmoid(types) > 0.5).int()
        for i in range(types.size(0)):
            if types[i].sum() == 0:
                types[i, torch.randint(0, self.cfg["type_dim"], (1,))] = 1

        talents = talents.argmax(dim=1)

        print_pokemon_entries(
                stats,
                types,
                talents,
                self.idx_to_type,
                self.idx_to_talent,
                self.cfg,
                )

def main():
    with open("pokemons.json", encoding="utf-8") as f:
        pokemons = json.load(f)

    model = PokeVAEGenerator.from_checkpoint("model.pt")

    print("Random:")
    model.sample_random(n=15)

if __name__ == "__main__":
    main()
