import os
import torch
from torch.utils import data
from PIL import Image
from torchvision import transforms


class get_data(data.Dataset):
    def __init__(self, img_root, gt_root, img_size, max_num=None, is_train=False):
        class_list = os.listdir(img_root)
        self.size = [img_size, img_size]
        self.img_dirs = list(map(lambda x: os.path.join(img_root, x), class_list))
        self.gt_dirs = list(map(lambda x: os.path.join(gt_root, x), class_list))
        self.max_num = max_num
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        img_names = os.listdir(self.img_dirs[item])
        gt_names = os.listdir(self.gt_dirs[item])
        num = len(img_names)
        img_paths = list(map(lambda x: os.path.join(self.img_dirs[item], x), img_names))
        gt_paths = list(map(lambda x: os.path.join(self.gt_dirs[item], x[:-4] + '.png'), gt_names))
        imgs = torch.Tensor(num, 3, self.size[0], self.size[1])
        gts = torch.Tensor(num, 1, self.size[0], self.size[1])
        subpaths = []
        ori_sizes = []
        for idx in range(num):
            image = Image.open(img_paths[idx]).convert('RGB')
            gt = Image.open(gt_paths[idx]).convert('L')
            subpaths.append(os.path.join(img_paths[idx].split('/')[-2], img_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((image.size[1], image.size[0]))
            image, gt = self.transform(image), self.transform(gt)
            imgs[idx] = image
            gts[idx] = gt
        return imgs, gts, subpaths, ori_sizes

    def __len__(self):
        return len(self.img_dirs)


def get_dataloader(img_root, gt_root, img_size, batch_size, shuffle=False, num_workers=0, pin=False):
    dataset = get_data(img_root, gt_root, img_size)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader
