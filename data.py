import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def encode_pokemon(pokemon, cfg, scaler, type_to_idx, talent_to_idx, model, device="cpu"):
    stats = np.array([[pokemon["base"][s] for s in cfg["stats_order"]]], dtype=np.float32)
    stats = scaler.transform(stats)

    types = encode_types(pokemon, type_to_idx)[None, :]
    talent_idx = encode_talent(pokemon, talent_to_idx)
    talents = np.zeros((1, cfg["talent_dim"]), dtype=np.float32)
    talents[0, talent_idx] = 1.0

    x = np.concatenate([stats, types, talents], axis=1)
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)

    return z, pokemon

def load_checkpoint(path):
    return torch.load(path, map_location="cpu")

def sample_random(model, cfg, scaler, idx_to_type, idx_to_talent, n=151):
    z = torch.randn(n, cfg["latent_dim"])
    stats, types, talents = model.decode(z)

    stats = scaler.inverse_transform(stats.numpy())
    types = (torch.sigmoid(types) > 0.5).int()
    for i in range(types.size(0)):
        if types[i].sum() == 0:
            types[i, torch.randint(0, cfg["type_dim"], (1,))] = 1

    talents = talents.argmax(dim=1)
    print_pokemon_entries(stats, types, talents, idx_to_type, idx_to_talent, cfg)

def encode_types(pokemon, type_to_idx):
    v = np.zeros(len(type_to_idx), dtype=np.float32)
    for t in pokemon["type"]:
        v[type_to_idx[t]] = 1.0
    return v

def encode_talent(pokemon, talent_to_idx):
    return talent_to_idx[pokemon["profile"]["ability"][0][0]]

def load_dataset(cfg):
    with open(cfg["pokemon_json"], encoding="utf-8") as f:
        pokemons = json.load(f)

    stats_order = cfg["stats_order"]

    all_types = sorted({t for p in pokemons for t in p["type"]})
    all_talents = sorted({t[0] for p in pokemons for t in p["profile"]["ability"]})

    type_to_idx = {t: i for i, t in enumerate(all_types)}
    talent_to_idx = {t: i for i, t in enumerate(all_talents)}

    X_stats = np.array([[p["base"][s] for s in stats_order] for p in pokemons], dtype=np.float32)
    scaler = StandardScaler()
    X_stats = scaler.fit_transform(X_stats)

    X_types = np.array([encode_types(p, type_to_idx) for p in pokemons], dtype=np.float32)

    Y = np.array([encode_talent(p, talent_to_idx) for p in pokemons])
    X_talents = np.zeros((len(pokemons), len(all_talents)), dtype=np.float32)
    X_talents[np.arange(len(Y)), Y] = 1.0

    X = np.concatenate([X_stats, X_types, X_talents], axis=1)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))

    return dataset, scaler, type_to_idx, talent_to_idx, pokemons

