import torch
from loss import vae_loss
import torch.optim as optim
from autoencoder import AutoEncoder
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision import datasets, transforms
from utils import AverageMeter, ProgressMeter


def predict(model, device, epoch):
    z = torch.randn(64, 2).to(device)
    # print(z)
    model.eval()
    with torch.no_grad():
        output = model.decoder(z).view(-1, 1, 28, 28)
        save_image(output, './Autoencoder/log/sample_{}.png'.format(epoch))


def validate(test_loader, model, device):
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.to(device)
            _, C, W, H = input.shape
            input = input.view((-1, W*C*H))
            output = model(input)
            x_decoded_mean, z_mean, z_log_var, z = output
            loss = vae_loss(input, x_decoded_mean, z_mean, z_log_var)
            losses.update(loss.item(), input.size(0))
        print(losses)
        return losses


def train(train_loader, model, optimizer, epoch, device):
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), losses)
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.to(device)
        _, C, W, H = input.shape
        input = input.view((-1, W*C*H))
        output = model(input)
        x_decoded_mean, z_mean, z_log_var, z = output
        optimizer.zero_grad()
        loss = vae_loss(input, x_decoded_mean, z_mean, z_log_var)
        losses.update(loss.item(), input.size(0))
        loss.backward()
        optimizer.step()
        print(losses)
        if i % 10 == 0:
            progress.pr2int(i)
    return losses


def main():
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize(mean=[0.5],
    #                                                     std=[0.5])])
    transform = transforms.Compose([transforms.ToTensor()])
    data_train = datasets.MNIST("./data/", transform=transform,
                                train=True,
                                download=False)
    data_test = datasets.MNIST("./data/", transform=transform,
                               train=False,
                               download=False)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=64,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=64,
                                              shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter()
    model = AutoEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # sheduler = optim.lr_scheduler.StepLR(optimizer, 40, 0.1)

    best_loss = 1000.0
    for epoch in range(150):
        print('Epoch: ', epoch)
        # sheduler.step(epoch)
        # predict(model, device, epoch)
        train_loss = train(train_loader, model, optimizer, epoch, device)
        val_loss = validate(test_loader, model, device)

        writer.add_scalar("scalar/train_loss", train_loss.avg, epoch)
        writer.add_scalar("scalar/val_loss", val_loss.avg, epoch)
        # writer.add_scalar("scalar/learing_rate", sheduler.get_lr()[0], epoch)

        for name, layer in model.named_parameters():
            writer.add_histogram(name+"_grad", layer.grad.cpu().data.numpy(), epoch)
            writer.add_histogram(name+"_data", layer.cpu().data.numpy(), epoch)

        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            torch.save(model.state_dict(), './autoencoder/best.pt')

    writer.close()
    print("finish")


if __name__ == "__main__":
    main()
