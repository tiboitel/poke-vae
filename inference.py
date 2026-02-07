import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import VAE
from data import *
from utils import print_pokemon_entries

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt

def sample_random(model, cfg, scaler, idx_to_type, idx_to_talent):
    z = torch.randn(1, cfg["latent_dim"])
    stats, types, talents = model.decode(z)

    stats = scaler.inverse_transform(stats.detach().numpy())
    types = (torch.sigmoid(types) > 0.5).int()
    for i in range(types.size(0)):
        if types[i].sum() == 0:
            types[i, torch.randint(0, cfg["type_dim"], (1,))] = 1

    talents = talents.argmax(dim=1)
    print_pokemon_entries(stats, types, talents, idx_to_type, idx_to_talent, cfg)

def sample_alternative_regional_form(
    model,
    pokemon,
    cfg,
    scaler,
    type_to_idx,
    talent_to_idx,
    idx_to_type,
    idx_to_talent,
    device="cpu",
    stat_noise=0.4
):
    z, base_pokemon = encode_pokemon(
        pokemon,
        cfg,
        scaler,
        type_to_idx,
        talent_to_idx,
        model,
        device
    )

    with torch.no_grad():
        # z = z + stat_noise * torch.randn_like(z)
        stats, types, talents = model.decode(z)

    stats = scaler.inverse_transform(stats.cpu().numpy())
    stats = np.clip(stats, 1, None)

    base_bst = sum(base_pokemon["base"][s] for s in cfg["stats_order"])
    new_bst = stats.sum()
    stats *= base_bst / max(new_bst, 1e-6)

    types = (torch.sigmoid(types) > 0.5).int()
    if types.sum() == 0:
        types[0, torch.randint(0, cfg["type_dim"], (1,))] = 1

    talents = talents.argmax(dim=1)

    print_pokemon_entries(stats, types, talents, idx_to_type, idx_to_talent, cfg)

def main():
    ckpt = load_checkpoint("model.pt")
    cfg = ckpt["config"]
    
    dataset, scaler, type_to_idx, talent_to_idx, pokemons = load_dataset(cfg)

    model = VAE(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"].detach().numpy()
    scaler.scale_ = ckpt["scaler_scale"].detach().numpy()
    scaler.n_features_in_ = scaler.mean_.shape[0]
    idx_to_type = {i: t for t, i in ckpt["type_to_idx"].items()}
    idx_to_talent = {i: t for t, i in ckpt["talent_to_idx"].items()}

    sample_random(model, cfg, scaler, idx_to_type, idx_to_talent)
    for i in range(151):
        print(f"""Name: {pokemons[i]["name"]["english"]}. Stats: """)
        sample_alternative_regional_form(
                model,
                pokemons[i],
                cfg=cfg,
                scaler=scaler,
                type_to_idx=ckpt["type_to_idx"],
                talent_to_idx=ckpt["talent_to_idx"],
                idx_to_type=idx_to_type,
                idx_to_talent=idx_to_talent
            )
        print()

if __name__ == "__main__":
    main()
