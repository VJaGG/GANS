import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, original_dim=28*28, intermediate_dim=256, latent_dim=2):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(original_dim, intermediate_dim)
        self.z_mean = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var = nn.Linear(intermediate_dim, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.encoder(x)
        h = F.relu(h)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        epsilon = torch.randn((batch_size, self.latent_dim)).cuda()
        z = z_mean + torch.exp(z_log_var) * epsilon
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    def __init__(self, original_dim=28*28, intermediate_dim=256, latent_dim=2):
        super(Decoder, self).__init__()
        self.decoder_h = nn.Linear(latent_dim, intermediate_dim)
        self.x_decoded = nn.Linear(intermediate_dim, original_dim)

    def forward(self, x):
        decoder_h = self.decoder_h(x)
        decoder_h = F.relu(decoder_h)
        x_decoded = self.x_decoded(decoder_h)
        x_decoded = torch.sigmoid(x_decoded)
        return x_decoded


class AutoEncoder(nn.Module):
    def __init__(self, original_dim=28*28, intermediate_dim=256, latent_dim=2):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(original_dim, intermediate_dim, latent_dim)
        self.decoder = Decoder(original_dim, intermediate_dim, latent_dim)

    def forward(self, input):
        z_mean, z_log_var, z = self.encoder(input)
        x = self.decoder(z)
        return x, z_mean, z_log_var, z


if __name__ == "__main__":
    encoder = Encoder()
    x = torch.rand(28*28)
    z_mean, z_log_var, z = encoder.forward(x)
    print(z)
    decoder = Decoder()
    x_decoder = decoder.forward(z)
    print(x_decoder.shape)
    model = AutoEncoder()
    x, z_mean, z_log_var, z = model(x)
    print(x.shape)
