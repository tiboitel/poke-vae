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

        return z, mu, logvar

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

    def sample_regional_form(
    self,
    pokemon: dict,
    temperature: float = 0.8,
    max_tries: int = 256,
    ):
        original_types = set(self.type_to_idx[t] for t in pokemon["type"])
        base_bst = sum(pokemon["base"][s] for s in self.cfg["stats_order"])

        with torch.no_grad():
            _, mu, logvar = self.encode_pokemon(pokemon)
            std = torch.exp(0.5 * logvar)

        best = None
        best_score = -1e9

        for _ in range(max_tries):
            with torch.no_grad():
                # --- latent sampling ---
                eps = torch.randn_like(std)
                z = mu + eps * std * temperature

                stats_pred, type_logits, talent_logits = self.model.decode(z)

                # --- TYPE SAMPLING (stochastic) ---
                type_probs = torch.sigmoid(type_logits)[0].cpu().numpy()

                sampled = set()
                for i, p in enumerate(type_probs):
                    if np.random.rand() < p:
                        sampled.add(i)

                # Repair: enforce 1–2 types
                if len(sampled) == 0:
                    sampled.add(np.argmax(type_probs))
                if len(sampled) > 2:
                    sampled = set(
                        sorted(sampled, key=lambda i: type_probs[i], reverse=True)[:2]
                    )

                if sampled == original_types:
                    continue

                # --- STATS ---
                stats = self.scaler.inverse_transform(
                    stats_pred.cpu().numpy()
                ).reshape(-1)

                stats = np.clip(stats, 1, None)
                stats *= base_bst / max(stats.sum(), 1e-6)
                stats = np.round(stats).astype(int)
                stats = np.clip(stats, 1, None)

                # --- TALENT CONFIDENCE ---
                logits = talent_logits[0]
                probs = torch.softmax(logits / 1.2, dim=0).cpu().numpy()

                top2 = np.argsort(probs)[-2:][::-1]
                best_idx, second_idx = top2

                margin = probs[best_idx] - probs[second_idx]
                entropy = -np.sum(probs * np.log(probs + 1e-9))

                if margin < 0.8:
                    continue

                type_conf = np.mean([type_probs[i] for i in sampled])
                score = margin + 0.3 * type_conf - 0.05 * entropy

                if score > best_score:
                    best_score = score
                    best = (stats, sampled, best_idx)

            if best is None:
                raise RuntimeError("Failed to sample a regional form")

            stats, sampled_types, talent_idx = best

            types_bin = np.zeros(len(self.type_to_idx), dtype=int)
            for t in sampled_types:
                types_bin[t] = 1

            print_pokemon_entries(
                stats.reshape(1, -1), 
                types_bin.reshape(1, -1),
                np.array([talent_idx]),
                self.idx_to_type,
                self.idx_to_talent,
                self.cfg,
            )

    def sample_fusion(self, pokemon1, pokemon2, stat_noise: float = 0.5):
        z1, _, _ = self.encode_pokemon(pokemon1)
        z2, _, _ = self.encode_pokemon(pokemon2)

        with torch.no_grad():
            z = (z1 + z2) / 2
            z = z + stat_noise * torch.randn_like(z)
            stats, _, _ = self.model.decode(z)

        stats = self.scaler.inverse_transform(stats.cpu().numpy())
        stats = np.clip(stats, 1, None)

        bst1 = sum(pokemon1["base"][s] for s in self.cfg["stats_order"])
        bst2 = sum(pokemon2["base"][s] for s in self.cfg["stats_order"])
        target_bst = round((bst1 + bst2) / 2)

        stats *= target_bst / max(stats.sum(), 1e-6)

        parent_types = list(
            {
                *[self.type_to_idx[t] for t in pokemon1["type"]],
                *[self.type_to_idx[t] for t in pokemon2["type"]],
            }
        )

        chosen = np.random.choice(parent_types, min(2, len(parent_types)), replace=False)
        types = torch.zeros((1, self.cfg["type_dim"]), dtype=torch.int)
        types[0, chosen] = 1

        parent_abilities = list(
            {
                *[self.talent_to_idx[a[0]] for a in pokemon1["profile"]["ability"]],
                *[self.talent_to_idx[a[0]] for a in pokemon2["profile"]["ability"]],
            }
        )

        talents = torch.tensor(
            [np.random.choice(parent_abilities)],
            dtype=torch.long,
        )

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
    model.sample_random()

    idx = 21
    print(f"\nRegional form of {pokemons[idx]['name']['english']}:")
    model.sample_regional_form(pokemons[idx])

    print("\nFusion:")
    model.sample_fusion(pokemons[241], pokemons[385])


if __name__ == "__main__":
    main()
