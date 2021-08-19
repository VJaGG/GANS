import os
import csv
import torch
from PIL import Image
import torch.utils.data as data
from torchvision.utils import make_grid
import torchvision.transforms as transforms


class Cartoon(data.Dataset):
    def __init__(self, root,  transform=None):
        super(Cartoon, self).__init__()
        self.root = root
        self.image_path = os.path.join(self.root, 'images')
        tags_path = os.path.join(self.root, "tags.csv")
        self.img_info = []
        classes_hair = set()
        classes_eyes = set()
        with open(tags_path, 'r') as f:
            lines = csv.reader(f)
            for row in lines:
                name = row[0]
                info = row[1].split(" ")
                hair_color = info[0]
                eye_color = info[2]
                classes_hair.add(hair_color)
                classes_eyes.add(eye_color)
                self.img_info.append([name, hair_color, eye_color])
        self.eyecolor_to_index = dict(zip(sorted(classes_eyes),
                                      range(len(classes_eyes))))
        self.haircolor_to_index = dict(zip(sorted(classes_hair),
                                           range(len(classes_hair))))
        print(self.haircolor_to_index)
        print(self.eyecolor_to_index)
        self.num_hair = len(classes_hair)
        self.num_eyes = len(classes_eyes)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img_info = self.img_info[index]
        img_name = img_info[0]+'.jpg'
        hair_classe = img_info[1]
        eye_classe = img_info[2]
        img_path = os.path.join(self.image_path, img_name)
        img = Image.open(img_path).convert('RGB')
        hair_label = self.haircolor_to_index[hair_classe]
        eye_label = self.eyecolor_to_index[eye_classe]
        if self.transform is not None:
            img = self.transform(img)
        return img, hair_label, eye_label

    def __len__(self):
        return len(self.img_info)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def make_image(tensor, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda img: img * 2.0 - 1.0, ])
    train_data = Cartoon("./data/extra_data", transform)
    print(len(train_data))
    img, hair_label, eye_label = train_data[10000]
    print(hair_label)
    print(eye_label)