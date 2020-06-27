import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
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




class VOCDataset(torch.utils.data.Dataset):
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                   'train', 'tvmonitor')

    def __init__(self,
                 data_dir,
                 split='train',
                 transform=None,
                 keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main",
                                       "%s.txt" % self.split)
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.batch_count = 0
        self.img_size = 416
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.class_dict = {
            class_name: i
            for i, class_name in enumerate(self.class_names)
        }

    def resize(self, img, size):
        image = F.interpolate(img.unsqueeze(0), size=size,
                              mode="nearest").squeeze(0)
        return image

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        targets = dict(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations",
                                       "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(
                int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes,
                         dtype=np.float32), np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations",
                                       "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(
            map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages",
                                  "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        if self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))
        self.batch_count += 1
        targets = torch.cat(targets, 0)
        imgs = torch.stack([self.resize(img, self.img_size) for img in imgs])
        return imgs, targets


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
                [ToPILImage(),
                 ColorJitter(0.2, 0.2, 0.2),
                 ToTensor()])
        else:
            self.transform = ToTensor()
        self.labels = {}
        with open(f"{root}/{mode}.txt", "r") as f:
            lines = f.readlines()
            self.lines = []
            for line in lines:
                row = line.strip()
                label_path = row.replace("JPEGImages",
                                         "labels").replace(".jpg", ".txt")
                data = np.loadtxt(label_path).reshape(-1, 5)
                if data.size == 0:
                    continue
                self.labels[row] = data
                self.lines.append(row)

    def resize(self, img, size):
        image = F.interpolate(img.unsqueeze(0), size=size,
                              mode="nearest").squeeze(0)
        return image

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        row = self.lines[index].strip()
        image = cv2.imread(row)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = self.transform(image)
        boxes = self.labels[row]
        if self.rtn_path:
            return row, img
        else:
            _, h, w = img.shape
            h_factor, w_factor = (h, w)
            img, pad = pad_to_square(img, 0)
            _, padded_h, padded_w = img.shape
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
            targets[:, 1:] = torch.FloatTensor(boxes)
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)
            return img, targets
