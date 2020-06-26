import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, Compose, ToPILImage, ToTensor


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


class VOCDataset(Dataset):
    def __init__(self, root, mode="train", img_size=608, rtn_path=False):
        self.mode = mode
        self.rtn_path = rtn_path

        self.img_size = img_size
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        if self.mode == "train":
            self.transform = Compose(
                [ToPILImage(), ColorJitter(0.2, 0.2, 0.2), ToTensor()]
            )
        else:
            self.transform = ToTensor()
        self.labels = {}
        with open(f"{root}/{mode}.txt", "r") as f:
            lines = f.readlines()
            self.lines = []
            for line in lines:
                row = line.strip()
                label_path = row.replace("JPEGImages", "labels").replace(".jpg", ".txt")
                data = np.loadtxt(label_path).reshape(-1, 5)
                if data.size == 0:
                    continue
                self.labels[row] = data
                self.lines.append(row)

    def resize(self, img, size):
        image = F.interpolate(img.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return image

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        row = self.lines[index].strip()
        image = cv2.imread(row)
        label = self.labels[row]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = self.transform(image)
        boxes = self.labels[row]
        if self.rtn_path:
            return row, img
        else:
            _, h, w = img.shape
            h_factor, w_factor = (h, w)
            img, pad = pad_to_square(img, 0)
            _, padded_n, padded_w = img.shape
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
            if np.random().random() < 0.5:
                img, targets = horisontal_flip(img, targets)
            return image, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        if self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        self.batch_count += 1
        targets = torch.cat(targets, 0)
        imgs = torch.stack([self.resize(img, self.img_size) for img in imgs])
        return imgs, targets
