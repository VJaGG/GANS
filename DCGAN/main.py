import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from dcgan import Generator, Discriminator
from torchvision.utils import save_image
from torchvision import datasets, transforms
sys.path.append("/data/bitt/wzq/wzq/GANs/")
from utils import AverageMeter, ProgressMeter


def train(train_loader, generator, discriminator, g_optimizer,
          d_optimizer, criteria, epoch, device):
    g_losses = AverageMeter('Generator Losses', ':.4e')
    d_losses = AverageMeter('Discriminator Losses', ':.4e')
    progress = ProgressMeter(len(train_loader), g_losses, d_losses)
    for i, (input, _) in enumerate(train_loader):
        input = input.to(device)
        batch_size, C, H, W = input.shape

        # update discriminator
        z = torch.randn(batch_size, 100).to(device)
        gen_imgs = generator(z)
        logits_fake = discriminator(gen_imgs)
        logits_real = discriminator(input)
        target_real = torch.ones_like(logits_real)
        target_fake = torch.zeros_like(logits_fake)

        d_optimizer.zero_grad()
        d_loss = criteria(logits_fake, target_fake) + criteria(logits_real,
                                                               target_real)
        d_losses.update(d_loss.item(), batch_size)
        d_loss.backward()
        d_optimizer.step()

        # update generator
        z = torch.randn(batch_size, 100).to(device)
        gen_imgs = generator(z)
        logits_fake = discriminator(gen_imgs)
        target_fake = torch.ones_like(logits_fake)

        g_optimizer.zero_grad()
        g_loss = criteria(logits_fake, target_fake)
        g_losses.update(g_loss.item(), batch_size)
        g_loss.backward()
        g_optimizer.step()

        # print log
        if i % 10 == 0:
            progress.pr2int(i)
    return d_losses, g_losses


def predict(generator, discriminator, device, criteria, epoch):
    z = torch.randn(64, 100).to(device)
    generator.eval()
    with torch.no_grad():
        output = generator(z).view(-1, 1, 28, 28)
        logits_fake = discriminator(output)
        target_fake = torch.zeros_like(logits_fake)
        val_loss = criteria(logits_fake, target_fake)
        img = (output + 1.0) / 2.0  # output是[-1, 1]之间的，归一化到[0.0, 1.0]之间
        save_image(img, './DCGAN/log/sample_{}.png'.format(epoch))
    return val_loss


def main():
    # transform = transforms.Compose([transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor(),  # [0.0, 1.0]
                                    lambda img: img * 2.0 - 1.0])  # [-1.0, 1.0]
    data_train = datasets.MNIST("./data/", transform=transform,
                                train=True,
                                download=False)
    # data_test = datasets.MNIST("./data/", transform=transform,
    #                            train=False,
    #                            download=False)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=64,
                                               shuffle=True)
    # test_loader = torch.utils.data.DataLoader(data_test, batch_size=64,
    #                                           shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    criteria = nn.BCELoss()
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = optim.Adam(generator.parameters(), 1e-3)
    d_optimizer = optim.Adam(discriminator.parameters(), 1e-3)
    min_loss = 100.0
    for epoch in range(150):
        print("Epoch: ", epoch)
        val_loss = predict(generator, discriminator, device, criteria, epoch)
        writer.add_scalar("val_loss", val_loss.detach(), epoch)
        d_losses, g_losses = train(train_loader, generator, discriminator,
                                   g_optimizer, d_optimizer, criteria,
                                   epoch, device)
        writer.add_scalars("loss", {"d_loss": d_losses.avg,
                                    "g_loss": g_losses.avg}, epoch)
        if val_loss.item() < min_loss:
            min_loss = val_loss.item()
            torch.save(generator.state_dict(),
                       'DCGAN/checkpoints/generator_best.pt')
            torch.save(discriminator.state_dict(),
                       'DCGAN/checkpoints/discriminator_best.pt')
    writer.close()
    print("finish")


if __name__ == "__main__":
    main()
