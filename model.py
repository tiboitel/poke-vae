import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        input_dim = cfg["stats_dim"] + cfg["type_dim"] + cfg["talent_dim"]

        self.encoder = nn.Sequential(
                nn.Linear(input_dim, cfg["hidden_encoder_dim"]),
                nn.ReLU(),
                )
        self.fc_mu = nn.Linear(cfg["hidden_encoder_dim"], cfg["latent_dim"])
        self.fc_logvar = nn.Linear(cfg["hidden_encoder_dim"], cfg["latent_dim"])

        self.decoder = nn.Sequential(
                nn.Linear(cfg["latent_dim"], cfg["hidden_decoder_dims"][0]),
                nn.ReLU(),
                nn.Dropout(cfg["dropout"]),
                nn.Linear(cfg["hidden_decoder_dims"][0], cfg["hidden_decoder_dims"][1]),
                nn.ReLU(),
                )

        self.stats_head = nn.Linear(cfg["hidden_decoder_dims"][1], cfg["stats_dim"])
        self.type_head = nn.Linear(cfg["hidden_decoder_dims"][1], cfg["type_dim"])
        self.talent_head = nn.Linear(cfg["hidden_decoder_dims"][1], cfg["talent_dim"])

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return (
                self.stats_head(h),
                self.type_head(h),
                self.talent_head(h),
                )

    def forward(self, x):
        s = self.cfg["stats_dim"]
        t = self.cfg["type_dim"]
        stats_gt = x[:, :s]
        types_gt = x[:, s:s + t]
        talents_gt = x[:, -self.cfg["talent_dim"]:]

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        stats, types, talents = self.decode(z)

        return stats, types, talents, stats_gt, types_gt, talents_gt, mu, logvar

