import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_embeddings=self.num_classes,
                                      embedding_dim=self.z_dim)
        self.fc1 = nn.Linear(z_dim, 256 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(256, 128, 3, 2,
                                        padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 1, 3, 2,
                                        padding=1, output_padding=1)

    def forward(self, z, label):
        label_embedding = self.embedding(label)
        joined_representation = z * label_embedding
        x = self.fc1(joined_representation)
        x = F.leaky_relu(x)
        x = x.view((-1, 256, 7, 7))
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, image_dim=28*28):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.image_dim = image_dim
        self.embedding = nn.Embedding(num_embeddings=self.num_classes,
                                      embedding_dim=self.image_dim)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.classifier = nn.Linear(1152, 1)

    def forward(self, img, label):
        batch_size, C, H, W = img.shape
        label_embedding = self.embedding(label)
        label_embedding = label_embedding.view(-1, C, H, W)
        x = torch.cat([img, label_embedding], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    label = torch.tensor([1, 2])
    z = z = torch.randn(2, 100)
    generator = Generator()
    output = generator(z, label)
    print(output.shape)
    discriminator = Discriminator()
    pred = discriminator(output, label)
    print(pred.shape)
