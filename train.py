import json
import torch
from torch.utils.data import DataLoader
from model import VAE
from loss import vae_loss
from data import load_dataset

def main():
    cfg = json.load(open("config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, scaler, type_to_idx, talent_to_idx, _ = load_dataset(cfg)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = VAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    
    print("--- Start training PokeVAE. ---")
    for epoch in range(cfg["epochs"]):
        total = 0.0
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            loss = vae_loss(model(x, cfg), epoch, cfg)
            loss.backward()
            opt.step()
            total += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss {total:.2f}")

    torch.save({
        "model": model.state_dict(),
        "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
        "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
        "type_to_idx": type_to_idx,
        "talent_to_idx": talent_to_idx,
        "config": cfg
        }, cfg["checkpoint_path"])


if __name__ == "__main__":
    main()

