import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from cgan_cartoon import Generator, Discriminator
import torchvision.transforms as transforms
sys.path.append("/data/bitt/wzq/wzq/GANs")
from utils import Cartoon, AverageMeter, ProgressMeter


def train(train_loader, generator, discriminator,
          g_optimizer, d_optimizer, criteria, device):
    g_losses = AverageMeter('Generator Losses', ':.4e')
    d_losses = AverageMeter('Discriminator Losses', ':.4e')
    progress = ProgressMeter(len(train_loader), g_losses, d_losses)
    generator.train()
    discriminator.train()
    for i, (input, hair_label, eye_label) in enumerate(train_loader):
        bath_size, C, H, W = input.shape
        input = input.to(device)
        hair_label = hair_label.to(device)
        eye_label = eye_label.to(device)

        # update discriminator
        z = torch.randn(bath_size, 100).to(device)
        gen_imgs = generator(z, hair_label, eye_label)
        logits_fake = discriminator(gen_imgs, hair_label, eye_label)
        logits_real = discriminator(input, hair_label, eye_label)
        target_fake = torch.zeros_like(logits_fake)
        target_real = torch.ones_like(logits_real)
        d_loss = criteria(logits_fake, target_fake) + criteria(logits_real,
                                                               target_real)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        d_losses.update(d_loss.item(), bath_size)

        # update generator
        z = torch.randn(bath_size, 100).to(device)
        hair_label = np.random.randint(0, 12, size=(bath_size))
        eye_label = np.random.randint(0, 10, size=(bath_size))
        hair_label = torch.from_numpy(hair_label).to(device)
        eye_label = torch.from_numpy(eye_label).to(device)
        gen_imgs = generator(z, hair_label, eye_label)
        logits_fake = discriminator(gen_imgs, hair_label, eye_label)
        target_fake = torch.ones_like(logits_fake)
        g_loss = criteria(logits_fake, target_fake)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        g_losses.update(g_loss.item(), bath_size)

        if i % 10 == 0:
            progress.pr2int(i)

    return d_losses, g_losses


def predict(generator, discriminator, criteria, device, epoch):
    z = torch.randn(64, 100).to(device)
    hair_label = np.random.randint(0, 12, size=(64))
    eye_label = np.random.randint(0, 10, size=(64))
    hair_label = torch.from_numpy(hair_label).to(device)
    eye_label = torch.from_numpy(eye_label).to(device)
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        gen_imgs = generator(z, hair_label, eye_label)
        pred = discriminator(gen_imgs, hair_label, eye_label)
        label = torch.ones_like(pred)
        val_loss = criteria(pred, label).item()
        img = (gen_imgs + 1.0) / 2.0
        save_image(img,
                   '/data/bitt/wzq/wzq/GANs/CGAN/log_cartoon/sample_{}.png'.format(epoch))
    return val_loss


def main():
    root = "./data/extra_data"
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda img: img * 2.0 - 1.0, ])
    train_data = Cartoon(root, transform)
    train_loader = data.DataLoader(train_data, batch_size=64,
                                   shuffle=True, num_workers=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    writer = SummaryWriter()
    # 学习率为1e-3的时候过大出现问题
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)
    # g_sheduler = optim.lr_scheduler.StepLR(g_optimizer, 10, gamma=0.8)
    # d_sheduler = optim.lr_scheduler.StepLR(d_optimizer, 10, gamma=0.8)
    criteria = nn.BCELoss()
    for epoch in range(50):
        print("Epoch: ", epoch)
        # g_sheduler.step()
        # d_sheduler.step()
        val_loss = predict(generator, discriminator, criteria, device, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        d_losses, g_losses = train(train_loader, generator, discriminator,
                                   g_optimizer, d_optimizer, criteria, device)
        writer.add_scalars("train_loss", {"d_loss": d_losses.avg,
                                          "g_loss": g_losses.avg}, epoch)
        torch.save(generator.state_dict(),
                   "/data/bitt/wzq/wzq/GANs/CGAN/checkpoints/generator_cartoon_best.pt")
        torch.save(discriminator.state_dict(),
                   "/data/bitt/wzq/wzq/GANs/CGAN/checkpoints/discriminator_cartoon_best.pt")


if __name__ == "__main__":
    main()
