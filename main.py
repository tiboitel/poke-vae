import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import json
import numpy as np

LATENT_DIM = 8
STATS_DIM = 6
TYPE_DIM = 18
INPUT_DIM = STATS_DIM + TYPE_DIM

STATS_ORDER = [
    "HP",
    "Attack",
    "Defense",
    "Sp. Attack",
    "Sp. Defense",
    "Speed"
]

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(INPUT_DIM, 64)
        self.fc_mu = nn.Linear(64, LATENT_DIM)
        self.fc_logvar = nn.Linear(64, LATENT_DIM)

        # Decoder
        self.fc2 = nn.Sequential(
                    nn.Linear(LATENT_DIM + TYPE_DIM, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                    )
        self.fc3 = nn.Linear(64, STATS_DIM)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, types):
        z = torch.cat([z, types], dim=1)
        h = F.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x):
        stats = x[:, :STATS_DIM]
        types = x[:, STATS_DIM:]

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        recon_stats = self.decode(z, types)
        return recon_stats, stats, mu, logvar

def vae_loss(recon_x, x, mu, logvar, epoch):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    warmup_epochs = 200
    beta = min(1.0, epoch / warmup_epochs)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl_loss
    return loss

def encode_types(pokemon, type_to_idx, num_types):
    vec = np.zeros(num_types, dtype=np.float32)
    for t in pokemon["type"]:
        vec[type_to_idx[t]] = 1.0
    return vec

def make_type_vector(types, type_to_idx, num_types):
    v = np.zeros(num_types, dtype=np.float32)
    for t in types:
        v[type_to_idx[t]] = 1.0
    return torch.tensor(v).unsqueeze(0)

def main():
    with open("pokemons.json", "r", encoding="utf-8") as f:
        pokemons = json.load(f)

    ALL_TYPES = sorted({t for p in pokemons for t in p["type"]})
    type_to_idx = {t: i for i, t in enumerate(ALL_TYPES)}

    TYPE_DIM = len(ALL_TYPES)
    print(len(ALL_TYPES), ALL_TYPES)

    X = np.array(
        [[p["base"][stat] for stat in STATS_ORDER] for p in pokemons],
        dtype=np.float32
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    T = np.array(
        [encode_types(p, type_to_idx, len(ALL_TYPES)) for p in pokemons],
        dtype=np.float32
    )

    X_combined = np.concatenate([X_scaled, T], axis=1)

    tensor_X = torch.tensor(X_combined)
    dataset = TensorDataset(tensor_X)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(400):
        total_loss = 0
        for (x,) in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, stats, mu, logvar = model(x)
            loss = vae_loss(recon, stats, mu, logvar, epoch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss {total_loss:.2f}")

    model.eval()
    print("Random Generated:")
    print(" HP | Atk | Def | Spe. Atk | Spe. Def | Speed")
    with torch.no_grad():
        for _ in range(10):
            z = torch.randn(1, LATENT_DIM).to(device)
            type_vec = make_type_vector(
                    ["Fire", "Dragon"], type_to_idx, len(ALL_TYPES)
            ).to(device)
            fake_stats = model.decode(z, type_vec)
            fake_mons = scaler.inverse_transform(fake_stats.detach().cpu().numpy())
            print(*fake_mons[0].round(0), sep="|")


    with torch.no_grad():
        mus = []
        stats = []
        for (x,) in dataloader:
            mu, _ = model.encode(x.to(device))
            mus.append(mu.cpu())
            stats.append(x[:, :STATS_DIM].cpu())

    mus = torch.cat(mus)
    mu_mean = mu.mean(dim=0)
    mu_std = mu.std(dim=0)
    print("mu mean:", mu_mean)
    print("mu std :", mu_std)
    stats = torch.cat(stats)
    
    plt.scatter(mus[:, 2], mus[:, 3], c=stats[:, 5], cmap="viridis")
    plt.colorbar(label="Speed")

    plt.show()


if __name__ == "__main__":
    main()

