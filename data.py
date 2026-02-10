import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
import torch


# ---------------------------------------------------------------------
# Encoding helpers (pure numpy)
# ---------------------------------------------------------------------

def encode_types(pokemon, type_to_idx):
    v = np.zeros(len(type_to_idx), dtype=np.float32)
    for t in pokemon["type"]:
        v[type_to_idx[t]] = 1.0
    return v


def encode_talent(pokemon, talent_to_idx):
    return talent_to_idx[pokemon["profile"]["ability"][0][0]]


def encode_pokemon_features(pokemon, cfg, scaler, type_to_idx, talent_to_idx):
    """
    Encode a single Pokémon into a full input feature vector (numpy).
    No torch, no model.
    """
    stats = np.array(
        [pokemon["base"][s] for s in cfg["stats_order"]],
        dtype=np.float32,
    )
    stats = scaler.transform(stats[None, :])[0]

    types = encode_types(pokemon, type_to_idx)

    talent_idx = encode_talent(pokemon, talent_to_idx)
    talents = np.zeros(cfg["talent_dim"], dtype=np.float32)
    talents[talent_idx] = 1.0

    return np.concatenate([stats, types, talents], axis=0)


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_dataset(cfg):
    with open(cfg["pokemon_json"], encoding="utf-8") as f:
        pokemons = json.load(f)

    stats_order = cfg["stats_order"]

    # --- Build vocabularies ---
    all_types = sorted({t for p in pokemons for t in p["type"]})
    all_talents = sorted({a[0] for p in pokemons for a in p["profile"]["ability"]})

    type_to_idx = {t: i for i, t in enumerate(all_types)}
    talent_to_idx = {t: i for i, t in enumerate(all_talents)}

    # --- Fit scaler on stats ---
    X_stats = np.array(
        [[p["base"][s] for s in stats_order] for p in pokemons],
        dtype=np.float32,
    )

    scaler = StandardScaler()
    scaler.fit(X_stats)

    # --- Encode full dataset ---
    X = np.stack(
        [
            encode_pokemon_features(
                p, cfg, scaler, type_to_idx, talent_to_idx
            )
            for p in pokemons
        ],
        axis=0,
    )

    # Torch only at the very edge
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))

    return dataset, scaler, type_to_idx, talent_to_idx, pokemons
