import torch
import torch.nn.functional as F


def vae_loss(x, x_decoded_mean, z_log_var, z_mean, original_dim=28*28):
    xent_loss = original_dim * F.mse_loss(x_decoded_mean, x)
    kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean**2 - torch.exp(z_log_var))
    # 之前写的是torch.sum()，梯度反向传播存在问题，梯度过大几乎饱和，不能优化
    return xent_loss + kl_loss
