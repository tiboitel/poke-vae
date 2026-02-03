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
# VAE Model
# -----------------------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.fc1 = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, LATENT_DIM)
        self.fc_logvar = nn.Linear(128, LATENT_DIM)

        # Decoder for stats
        self.fc_stats = nn.Sequential(
            nn.Linear(LATENT_DIM + TYPE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, STATS_DIM)
        )

        # Decoder for talents
        self.fc_talent = nn.Sequential(
            nn.Linear(LATENT_DIM + TYPE_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, TALENT_DIM)
        )

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, types, type_scale=1.0):
        # Apply type warmup scaling
        types_scaled = types * type_scale
        x = torch.cat([z, types_scaled], dim=1)
        stats_out = self.fc_stats(x)
        talent_logits = self.fc_talent(x)
        return stats_out, talent_logits

    def forward(self, x, type_scale=1.0):
        stats = x[:, :STATS_DIM]
        types = x[:, STATS_DIM:STATS_DIM+TYPE_DIM]
        talents = x[:, -TALENT_DIM:]

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        stats_out, talent_logits = self.decode(z, types, type_scale)
        return stats_out, talent_logits, stats, talents, mu, logvar

# -----------------------------
# VAE Loss with KL warmup + free bits
# -----------------------------
def vae_loss(stats_out, talent_logits, stats, talents, mu, logvar, epoch, kl_warmup_epochs=400):
    stats_loss = F.mse_loss(stats_out, stats, reduction="mean")
    talent_loss = F.cross_entropy(talent_logits, talents.argmax(dim=1))

    # KL divergence
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=0.05)  # free bits per dimension
    kl_loss = torch.mean(torch.sum(kl_per_dim, dim=1))

    # KL warmup
    beta = min(1.0, epoch / kl_warmup_epochs)

    return stats_loss + 0.5 * talent_loss + beta * kl_loss

# -----------------------------
# Data helpers
# -----------------------------
def encode_types(pokemon, type_to_idx, num_types):
    vec = np.zeros(num_types, dtype=np.float32)
    for t in pokemon["type"]:
        vec[type_to_idx[t]] = 1.0
    return vec

def encode_talent(pokemon, talent_to_idx):
    return talent_to_idx[pokemon["profile"]["ability"][0][0]]

def make_type_vector(types, type_to_idx, num_types):
    v = np.zeros(num_types, dtype=np.float32)
    for t in types:
        v[type_to_idx[t]] = 1.0
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0)

# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    with open("pokemons.json", "r", encoding="utf-8") as f:
        pokemons = json.load(f)

    # Type and talent mappings
    ALL_TYPES = sorted({t for p in pokemons for t in p["type"]})
    type_to_idx = {t: i for i, t in enumerate(ALL_TYPES)}

    ALL_TALENTS = sorted({t[0] for p in pokemons for t in p["profile"]["ability"]})
    talent_to_idx = {t: i for i, t in enumerate(ALL_TALENTS)}

    # Stats normalization
    X_stats = np.array([[p["base"][stat] for stat in STATS_ORDER] for p in pokemons], dtype=np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stats)

    # One-hot types
    T = np.array([encode_types(p, type_to_idx, len(ALL_TYPES)) for p in pokemons], dtype=np.float32)

    # Talents one-hot
    Y = np.array([encode_talent(p, talent_to_idx) for p in pokemons], dtype=np.int64)
    TALENTS_ONEHOT = np.zeros((len(pokemons), len(ALL_TALENTS)), dtype=np.float32)
    for i, idx in enumerate(Y):
        TALENTS_ONEHOT[i, idx] = 1.0

    # Combine features
    X_combined = np.concatenate([X_scaled, T, TALENTS_ONEHOT], axis=1)
    tensor_X = torch.tensor(X_combined, dtype=torch.float32)
    dataset = TensorDataset(tensor_X)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model and optimizer
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    type_warmup_epochs = 200
    kl_warmup_epochs = 400
    for epoch in range(800):
        total_loss = 0
        # Gradually increase type contribution
        type_scale = min(1.0, epoch / type_warmup_epochs)

        for (x,) in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            stats_out, talent_logits, stats, talents, mu, logvar = model(x, type_scale=type_scale)
            loss = vae_loss(stats_out, talent_logits, stats, talents, mu, logvar, epoch, kl_warmup_epochs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 100 == 0:
            with torch.no_grad():
                mu, logvar = model.encode(tensor_X.to(device))
                z_std = torch.exp(0.5 * logvar)
                print(f"Epoch {epoch}. Latent std mean: {z_std.mean():.4f}. Total_loss: {total_loss:.2f}")

    # Random generation
    model.eval()
    print("\nRandom Generated Stats:")
    with torch.no_grad():
        z = torch.randn(5, LATENT_DIM).to(device)
        type_vec = make_type_vector(["Poison"], type_to_idx, len(ALL_TYPES)).to(device).repeat(5, 1)
        recon_stats, recon_talent = model.decode(z, type_vec, type_scale=1.0)
        recon_stats_real = scaler.inverse_transform(recon_stats.cpu().numpy())
        print(recon_stats_real)
        print("Generated talent indices:", recon_talent[:, :TALENT_DIM].argmax(dim=1))
        idx_to_talent = {idx: name for name, idx in talent_to_idx.items()}
        generated_indices = recon_talent[:, :TALENT_DIM].argmax(dim=1).cpu().numpy()
        generated_talent_names = [idx_to_talent[i] for i in generated_indices]
        print("Generated talents:", generated_talent_names)

    # Latent space stats
    with torch.no_grad():
        mus, logvars = [], []
        for x, in dataloader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
        mus = torch.cat(mus)
        logvars = torch.cat(logvars)
        print("\nLatent mean:", mus.mean(dim=0))
        print("Latent std :", mus.std(dim=0))
        print("Logvar mean:", logvars.mean(dim=0))

if __name__ == "__main__":
    main()
