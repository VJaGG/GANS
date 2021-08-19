import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256 * 8 * 8)
        self.conv1 = nn.ConvTranspose2d(256, 128, 3, 2,
                                        padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 3, 2,
                                        padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 3, 3, 2,
                                        padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256, 8, 8)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(1152, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    generator = Generator()
    z = torch.randn(64, 100)
    x = generator(z)
    discriminator = Discriminator()
    print(x.shape)
    pred = discriminator(x)
    print(pred.shape)
