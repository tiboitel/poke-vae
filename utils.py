import torch

def print_pokemon_entries(stats, types, talents, idx_to_type, idx_to_talent, cfg):
    order = cfg["stats_order"]
    for i in range(len(stats)):
        s = stats[i]
        t_idx = types[i].nonzero()[0].tolist()
        t_names = [idx_to_type[j] for j in t_idx]
        talent = idx_to_talent[talents[i].item()]

    type_str = " / ".join(t_names) if t_names else "Unknown"
    stats_str = " - ".join(f"{n}: {int(v)}" for n, v in zip(order, s))
    bst = round(sum(s))
    print(f"{stats_str} - BST: {bst} - {type_str} - {talent}")
