import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, Compose, ToPILImage, ToTensor


class VOCDataset(Dataset):
    def __init__(self, root, mode="train", img_size=416):
        self.mode = mode
        self.img_size = img_size
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
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        boxes = self.labels[row]
        target = torch.zeros((len(boxes), 6))
        target[:, 1:] = torch.from_numpy(boxes)
        return image, target

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        imgs = torch.stack([self.resize(img, self.img_size) for img in imgs])
        return imgs, targets
