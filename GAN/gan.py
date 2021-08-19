import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim=100, intermediate_dim=128, image_dim=28*28):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, image_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        output = torch.tanh(x)
        output = output.view(-1, 1, 28, 28)
        return output


class Discriminator(nn.Module):
    def __init__(self, z_dim=100, intermediate_dim=128, image_dim=28*28):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(image_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, 1)

    def forward(self, x):
        _, C, H, W = x.shape
        x = x.view(-1, C*H*W)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output


if __name__ == "__main__":
    x = torch.randn(100)
    generator = Generator()
    output = generator(x)
    print(output.shape)
    discriminator = Discriminator()
    prob = discriminator(output)
    print(prob)
