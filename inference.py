import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import VAE
from data import *
from utils import print_pokemon_entries

def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
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

def get_parent_ability_indices(pokemon, talent_to_idx):
    abilities = pokemon["profile"]["ability"]
    return [talent_to_idx[a[0]] for a in abilities]

def get_parent_type_indices(pokemon, type_to_idx):
    return [type_to_idx[t] for t in pokemon["type"]]

def sample_fusion_form(
    model,
    pokemon1,
    pokemon2,
    cfg,
    scaler,
    type_to_idx,
    talent_to_idx,
    idx_to_type,
    idx_to_talent,
    device="cpu"
):
    z1, base1 = encode_pokemon(
        pokemon1, cfg, scaler, type_to_idx, talent_to_idx, model, device
    )
    z2, base2 = encode_pokemon(
        pokemon2, cfg, scaler, type_to_idx, talent_to_idx, model, device
    )

    with torch.no_grad():
        z = (z1 + z2) / 2
        z = z + 0.4 * torch.randn_like(z)
        stats, _, _ = model.decode(z)  # ignore decoded types & abilities

    # ---------- Stats ----------
    stats = scaler.inverse_transform(stats.cpu().numpy())
    stats = np.clip(stats, 1, None)
    base_bst1 = sum(pokemon1["base"][s] for s in cfg["stats_order"])
    base_bst2 = sum(pokemon2["base"][s] for s in cfg["stats_order"])
    base_bst = round((base_bst1 + base_bst2) / 2)
    new_bst = stats.sum()
    stats *= base_bst / max(new_bst, 1e-6)

    # ---------- Types (parents only) ----------
    parent_types = (
        get_parent_type_indices(pokemon1, type_to_idx)
        + get_parent_type_indices(pokemon2, type_to_idx)
    )
    parent_types = list(set(parent_types))

    num_types = min(len(parent_types), np.random.choice([2, 2]))
    chosen_types = np.random.choice(parent_types, num_types, replace=False)

    types = torch.zeros((1, cfg["type_dim"]), dtype=torch.int)
    types[0, chosen_types] = 1

    # ---------- Abilities (parents only) ----------
    parent_abilities = (
        get_parent_ability_indices(pokemon1, talent_to_idx)
        + get_parent_ability_indices(pokemon2, talent_to_idx)
    )
    parent_abilities = list(set(parent_abilities))

    talents = torch.tensor(
        [np.random.choice(parent_abilities)],
        dtype=torch.long
    )

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
        z = z + stat_noise * torch.randn_like(z)
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

    print("""Random stats block:""")
    sample_random(model, cfg, scaler, idx_to_type, idx_to_talent)
    print()
    
    z1_idx =  241
    z2_idx = 385
    print(f"""Name: {pokemons[z2_idx]["name"]["english"]}. Stats: """)
    sample_alternative_regional_form(
            model,
            pokemons[z2_idx],
            cfg=cfg,
            scaler=scaler,
            type_to_idx=ckpt["type_to_idx"],
            talent_to_idx=ckpt["talent_to_idx"],
            idx_to_type=idx_to_type,
            idx_to_talent=idx_to_talent
        )
    print()

    print(f"""Name: {pokemons[z1_idx]["name"]["english"]}. Stats: """)
    sample_alternative_regional_form(
            model,
            pokemons[z1_idx],
            cfg=cfg,
            scaler=scaler,
            type_to_idx=ckpt["type_to_idx"],
            talent_to_idx=ckpt["talent_to_idx"],
            idx_to_type=idx_to_type,
            idx_to_talent=idx_to_talent
        )
    print()

    print(f"""Name: {pokemons[z2_idx]["name"]["english"]} x {pokemons[z1_idx]["name"]["english"]}. """)
    sample_fusion_form(model, pokemons[z2_idx], pokemons[z1_idx], cfg=cfg, scaler=scaler,
                type_to_idx=ckpt["type_to_idx"],
                talent_to_idx=ckpt["talent_to_idx"],
                idx_to_type=idx_to_type,
                idx_to_talent=idx_to_talent
            )


if __name__ == "__main__":
    main()
