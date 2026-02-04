import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Constants
# -----------------------------
LATENT_DIM = 64
STATS_DIM = 6
TYPE_DIM = 18
TALENT_DIM = 286
INPUT_DIM = STATS_DIM + TYPE_DIM + TALENT_DIM

STATS_ORDER = ["HP", "Attack", "Defense", "Sp. Attack", "Sp. Defense", "Speed"]

# -----------------------------
# VAE Model (Fix 4)
# -----------------------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, LATENT_DIM)
        self.fc_logvar = nn.Linear(128, LATENT_DIM)

        # Decoder trunk
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Heads
        self.stats_head = nn.Linear(128, STATS_DIM)
        self.type_head = nn.Linear(128, TYPE_DIM)       # multi-label
        self.talent_head = nn.Linear(128, TALENT_DIM)   # categorical

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        stats = self.stats_head(h)
        types = self.type_head(h)
        talents = self.talent_head(h)
        return stats, types, talents

    def forward(self, x):
        stats_gt = x[:, :STATS_DIM]
        types_gt = x[:, STATS_DIM:STATS_DIM + TYPE_DIM]
        talents_gt = x[:, -TALENT_DIM:]

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        stats, types, talents = self.decode(z)

        return stats, types, talents, stats_gt, types_gt, talents_gt, mu, logvar

# -----------------------------
# Loss
# -----------------------------
def vae_loss(stats, types, talents,
             stats_gt, types_gt, talents_gt,
             mu, logvar, epoch, kl_warmup_epochs=600):

    stats_loss = F.mse_loss(stats, stats_gt)

    type_loss = F.binary_cross_entropy_with_logits(
        types, types_gt, reduction="mean"
    )

    talent_loss = F.cross_entropy(
        talents, talents_gt.argmax(dim=1)
    )

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = torch.clamp(kl, min=0.02)
    kl = kl.sum(dim=1).mean()

    beta = min(0.02, epoch / kl_warmup_epochs)

    return stats_loss + type_loss + 0.5 * talent_loss + beta * kl

# -----------------------------
# Data helpers
# -----------------------------
def encode_types(pokemon, type_to_idx):
    v = np.zeros(len(type_to_idx), dtype=np.float32)
    for t in pokemon["type"]:
        v[type_to_idx[t]] = 1.0
    return v

def encode_talent(pokemon, talent_to_idx):
    return talent_to_idx[pokemon["profile"]["ability"][0][0]]

def print_pokemon_entries(stats, types, talents, idx_to_type, idx_to_talent):
    for i in range(len(stats)):
        s = stats[i]
        if torch.is_tensor(types):
            t_indices = types[i].nonzero(as_tuple=True)[0].cpu().numpy()
        else:
            t_indices = np.nonzero(types[i])[0]
        t_names = [idx_to_type[idx] for idx in t_indices]
        talent_name = idx_to_talent[talents[i].item()]

        if len(t_names) == 0:
            type_str = "Unknown"
        else:
            # Build type string
            type_str = t_names[0]
            if len(t_names) > 1:
                type_str += f" / {t_names[1]}"

        # Build stats string
        stats_str = " - ".join(f"{name}: {int(val)}" for name, val in zip(STATS_ORDER, s))
        bst = round(sum(stats[i]))
        print(f"{stats_str} - BST: {bst} - {type_str} - {talent_name}")

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("pokemons.json", "r", encoding="utf-8") as f:
        pokemons = json.load(f)

    ALL_TYPES = sorted({t for p in pokemons for t in p["type"]})
    type_to_idx = {t: i for i, t in enumerate(ALL_TYPES)}

    ALL_TALENTS = sorted({t[0] for p in pokemons for t in p["profile"]["ability"]})
    talent_to_idx = {t: i for i, t in enumerate(ALL_TALENTS)}

    # Stats
    X_stats = np.array([[p["base"][s] for s in STATS_ORDER] for p in pokemons], dtype=np.float32)
    scaler = StandardScaler()
    X_stats = scaler.fit_transform(X_stats)

    # Types
    X_types = np.array([encode_types(p, type_to_idx) for p in pokemons], dtype=np.float32)

    # Talents
    Y = np.array([encode_talent(p, talent_to_idx) for p in pokemons])
    X_talents = np.zeros((len(pokemons), len(ALL_TALENTS)), dtype=np.float32)
    for i, y in enumerate(Y):
        X_talents[i, y] = 1.0

    X = np.concatenate([X_stats, X_types, X_talents], axis=1)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # stats
    for epoch in range(400):
        total = 0
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            out = model(x)
            loss = vae_loss(*out, epoch, 200)
            loss.backward()
            opt.step()
            total += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss {total:.2f}")

    # -----------------------------
    # Random generation
    # -----------------------------
    model.eval()
    with torch.no_grad():
        z = torch.randn(151, LATENT_DIM).to(device)
        stats, types, talents = model.decode(z)

        stats = scaler.inverse_transform(stats.cpu().numpy())
        types = (torch.sigmoid(types) > 0.5).int()
        for i in range(types.size(0)):
            if types[i].sum() == 0:
                idx = torch.randint(0, TYPE_DIM, (1,))
                types[i, idx] = 1

        talents = talents.argmax(dim=1)

        print("\nGenerated stats:\n", stats)
        print("Generated types:\n", types.cpu().numpy())
        print("Generated talents:\n", talents.cpu().numpy())

        idx_to_type = {i: t for t, i in type_to_idx.items()}
        idx_to_talent = {i: t for t, i in talent_to_idx.items()}

        print("\nGenerated Pokémon entries:")
        print_pokemon_entries(stats, types, talents, idx_to_type, idx_to_talent)

        with torch.no_grad():
            for _ in range(10):
                pokemon = next(p for p in pokemons if p["name"]["english"] == "Nidoking")
                ### Pokemon encoding
                stats = np.array([[pokemon["base"][s] for s in STATS_ORDER]], dtype=np.float32)
                stats = scaler.transform(stats)

                # types
                types = encode_types(pokemon, type_to_idx)[None, :]

                # talent (one-hot)
                talent_idx = encode_talent(pokemon, talent_to_idx)
                talents = np.zeros((1, TALENT_DIM), dtype=np.float32)
                talents[0, talent_idx] = 1.0

                x = np.concatenate([stats, types, talents], axis=1)
                x = torch.tensor(x, dtype=torch.float32).to(device)

                with torch.no_grad():
                    mu, logvar = model.encode(x)
                    z = model.reparameterize(mu, logvar)
                    z = z + 0.8 * torch.randn_like(z)
                    stats_base, types_base, talents_base = model.decode(z)

                 # stats → real space
                stats_base_real = scaler.inverse_transform(stats_base.cpu().numpy())
                stats_base_real = np.clip(stats_base_real, 1, None)

                # types → binary
                types_base = (torch.sigmoid(types_base) > 0.5).int()
                for i in range(types_base.size(0)):
                    if types_base[i].sum() == 0:
                        idx = torch.randint(0, TYPE_DIM, (1,))
                        types_base[i, idx] = 1

                # talents → class index
                talents_base = talents_base.argmax(dim=1)

                print_pokemon_entries(
                    stats_base_real,
                    types_base,
                    talents_base,
                    idx_to_type,
                    idx_to_talent
                )
         
if __name__ == "__main__":
    main()
