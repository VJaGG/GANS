import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim=100, classes_hair=12, classes_eyes=10):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_classes_hair = classes_hair
        self.num_classes_eyes = classes_eyes
        self.hair_embedding = nn.Embedding(self.num_classes_hair, self.z_dim)
        self.eye_embedding = nn.Embedding(self.num_classes_eyes, self.z_dim)
        # self.fc1 = nn.Linear(self.z_dim*3, 256*7*7)
        self.fc1 = nn.Linear(self.z_dim, 256*7*7)
        self.conv1 = nn.Conv2d(256, 256*4, kernel_size=3, stride=1, padding=1)
        self.pixshuffle1 = nn.PixelShuffle(2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256*2, kernel_size=3, stride=1, padding=2)
        self.pixshuffle2 = nn.PixelShuffle(2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pixshuffle3 = nn.PixelShuffle(2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z, hair_lable, eye_lable):
        hair_embedding = self.hair_embedding(hair_lable)
        eye_embedding = self.eye_embedding(eye_lable)
        # z_dim = torch.cat((z, hair_embedding, eye_embedding), dim=1)
        z_dim = hair_embedding * eye_embedding * z
        x = self.fc1(z_dim)
        x = F.leaky_relu(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv1(x)
        x = self.pixshuffle1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.pixshuffle2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = self.pixshuffle3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = torch.tanh(x)
        return x


# class Generator(nn.Module):
#     def __init__(self, z_dim=100, classes_hair=12, classes_eyes=10):
#         super(Generator, self).__init__()
#         self.z_dim = z_dim
#         self.num_classes_hair = classes_hair
#         self.num_classes_eyes = classes_eyes
#         self.hair_embedding = nn.Embedding(self.num_classes_hair, self.z_dim)
#         self.eye_embedding = nn.Embedding(self.num_classes_eyes, self.z_dim)
#         self.fc1 = nn.Linear(self.z_dim*3, 256*7*7)

#         self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
#         self.bn1 = nn.BatchNorm2d(128)
#         self.conv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1,
#                                         padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2,
#                                         output_padding=1)

#     def forward(self, z, hair_lable, eye_lable):
#         hair_embedding = self.hair_embedding(hair_lable)
#         eye_embedding = self.eye_embedding(eye_lable)
#         z_dim = torch.cat((z, hair_embedding, eye_embedding), dim=1)
#         x = self.fc1(z_dim)
#         x = F.leaky_relu(x)
#         x = x.view(-1, 256, 7, 7)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = F.leaky_relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = F.leaky_relu(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = F.leaky_relu(x)
#         x = self.conv4(x)
#         x = torch.tanh(x)
#         return x


class Discriminator(nn.Module):
    def __init__(self, classes_hair=12, classes_eyes=10, image_dim=3*64*64, z_dim=100):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.image_dim = image_dim
        self.num_classes_hair = classes_hair
        self.num_classes_eyes = classes_eyes
        self.hair_embedding = nn.Embedding(self.num_classes_hair,
                                           self.z_dim)
        self.fc1 = nn.Linear(self.z_dim, self.image_dim)
        self.eye_embedding = nn.Embedding(self.num_classes_eyes,
                                          self.z_dim)
        self.fc2 = nn.Linear(self.z_dim, self.image_dim)
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.classifier = nn.Linear(1152, 1)

    def forward(self, img, hair_label, eye_label):
        batch_size, C, H, W = img.shape
        hair_embedding = self.hair_embedding(hair_label)
        hair_embedding = self.fc1(hair_embedding)
        eye_embedding = self.eye_embedding(eye_label)
        eye_embedding = self.fc2(eye_embedding)
        hair_embedding = hair_embedding.view(-1, C, H, W)
        eye_embedding = eye_embedding.view(-1, C, H, W)
        x = torch.cat((img, hair_embedding, eye_embedding), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
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
        x = self.classifier(x)
        output = torch.sigmoid(x)
        return output


if __name__ == "__main__":
    import numpy as np
    z = torch.randn(64, 100)
    hair_label = np.random.randint(0, 12, size=(64))
    print(hair_label.shape)
    eye_label = np.random.randint(0, 10, size=(64))
    hair_label = torch.from_numpy(hair_label)
    eye_label = torch.from_numpy(eye_label)
    # print(hair_label)
    # print(hair_label.shape)  # torch.Size([64, 1]) torch.Size([64])=torch.size([1, 64])
    # print(eye_label)
    generator = Generator()
    output = generator(z, hair_label, eye_label)
    discriminator = Discriminator()
    output = discriminator(output, hair_label, eye_label)
    print(output.shape)
    