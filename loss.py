import torch
import torch.nn.functional as F

def vae_loss(out, epoch, cfg):
    stats, types, talents, stats_gt, types_gt, talents_gt, mu, logvar = out

    stats_loss = F.mse_loss(stats, stats_gt)
    type_loss = F.binary_cross_entropy_with_logits(types, types_gt)
    talent_loss = F.cross_entropy(talents, talents_gt.argmax(dim=1))

    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl.mean(dim=0)
    kl_per_dim = torch.clamp(kl_per_dim, min=cfg["kl_free_bits"])
    kl = kl_per_dim.sum()

    beta = min(cfg["kl_max_beta"], epoch / cfg["kl_warmup_epochs"])

    total_loss = stats_loss + type_loss + cfg["talent_loss_weight"] * talent_loss + beta * kl
    return {
            "loss": total_loss,
            "stats_loss": stats_loss.detach(),
            "type_loss": type_loss.detach(),
            "talent_loss": talent_loss.detach(),
            "kl_raw_per_dim": kl_per_dim.detach(),
            "kl_loss": kl.detach()
    }
