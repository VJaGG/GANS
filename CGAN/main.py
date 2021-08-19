import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.datasets as datasets
from torchvision.utils import save_image
from cgan import Generator, Discriminator
import torchvision.transforms as transforms
sys.path.append("/data/bitt/wzq/wzq/GANs/")
from utils import AverageMeter, ProgressMeter


def validate(generator, discriminator, device, criteria, epoch):
    z = torch.randn(11, 100).to(device)
    label = torch.tensor([1, 8, 8, 6, 2, 3, 4, 4, 7, 0, 0]).to(device)
    # z = torch.randn(64, 100).to(device)
    # label = np.random.randint(0, 10, size=(64))
    # label = torch.from_numpy(label).to(device)
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        output = generator(z, label)
        logits_fake = discriminator(output, label)
        target_fake = torch.zeros_like(logits_fake)
        val_loss = criteria(logits_fake, target_fake)
        img = (output + 1.0) / 2.0  # output是[-1, 1]之间的，归一化到[0.0, 1.0]之间
        save_image(img, '/data/bitt/wzq/wzq/GANs/CGAN/log/sample_{}.png'.format(epoch), nrow=11)
        # save_image(img, '/data/bitt/wzq/wzq/GANs/CGAN/log/sample_{}.png'.format(epoch))
    return val_loss


def train(train_loader, generator, discriminator,
          g_optimizer, d_optimizer, criteria, device):
    g_losses = AverageMeter("Generator Losses", ":.4e")
    d_losses = AverageMeter("Discriminator Losses", ":.4e")
    progress = ProgressMeter(len(train_loader), g_losses, d_losses)
    for i, (input, target) in enumerate(train_loader):
        batch_size, C, H, W = input.shape
        input = input.to(device)

        # update discriminator
        z = torch.randn(batch_size, 100).to(device)
        label = target.to(device)
        gen_imgs = generator(z, label)
        logits_fake = discriminator(gen_imgs, label)
        logits_real = discriminator(input, label)
        target_real = torch.ones_like(logits_real)
        target_fake = torch.zeros_like(logits_fake)
        d_loss = criteria(logits_real, target_real) + criteria(logits_fake,
                                                               target_fake)
        d_losses.update(d_loss.item(), batch_size)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # update generator
        z = torch.randn(batch_size, 100).to(device)
        label = np.random.randint(0, 10, size=(batch_size))
        label = torch.from_numpy(label).to(device)
        gen_imgs = generator(z, label)
        logits_fake = discriminator(gen_imgs, label)
        target_fake = torch.ones_like(logits_fake)
        g_loss = criteria(logits_fake, target_fake)
        g_losses.update(g_loss.item(), batch_size)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if i % 10 == 0:
            progress.pr2int(i)
    return d_losses, g_losses


def main():
    transform = transforms.Compose([transforms.ToTensor(),  # [0.0, 1.0]
                                    lambda img: img * 2.0 - 1.0, ])  # [-1.0, 1.0]
    data_train = datasets.MNIST("./data/", transform=transform,
                                train=True,
                                download=False)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=64,
                                               shuffle=True, num_workers=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criteria = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    writer = SummaryWriter()
    # min_loss = 100.0
    for epoch in range(50):
        print("Epoch: ", epoch)
        val_loss = validate(generator, discriminator, device, criteria, epoch)
        writer.add_scalar("val_loss", val_loss.item(), epoch)
        d_loss, g_loss = train(train_loader, generator, discriminator,
                               g_optimizer, d_optimizer, criteria, device)
        writer.add_scalars("loss", {"d_loss": d_loss.avg,
                                    "g_loss": g_loss.avg}, epoch)
        # 不能保存验证损失最小的，判别器也是一直在变化的。
        # if val_loss.item() < min_loss:
        #     min_loss = val_loss.item()
        #     torch.save(generator.state_dict(),
        #                "/data/bitt/wzq/wzq/GANs/CGAN/checkpoints/generator_best.pt")
        #     torch.save(discriminator.state_dict(),
        #                "/data/bitt/wzq/wzq/GANs/CGAN/checkpoints/discriminator_best.pt")
        torch.save(generator.state_dict(),
                   "/data/bitt/wzq/wzq/GANs/CGAN/checkpoints/generator_best.pt")
        torch.save(discriminator.state_dict(),
                   "/data/bitt/wzq/wzq/GANs/CGAN/checkpoints/discriminator_best.pt")

if __name__ == "__main__":
    main()
