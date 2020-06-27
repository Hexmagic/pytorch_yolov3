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

from data.transforms import (ConvertFromInts, Expand, PhotometricDistort,
                             RandomMirror, RandomSampleCrop, Resize,
                             SubtractMeans, ToPercentCoords, ToTensor, Compose)


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
                 img_size=416,
                 keep_difficult=False):
        """Dataset for VOC data.
		Args:
			data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
				Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
		"""
        if split == 'train':
            transform = [
                ConvertFromInts(),
                PhotometricDistort(),
                Expand([123, 117, 104]),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                #Resize(img_size),
                SubtractMeans([123, 117, 104]),
                ToTensor(),
            ]
        else:
            transform = [
                Resize(img_size),
                SubtractMeans([123, 117, 104]),
                ToTensor()
            ]
        self.transform = Compose(transform)
        self.data_dir = data_dir
        self.split = split
        if split != 'test':
            image_sets_file = [
                os.path.join(self.data_dir, f'VOC{year}', "ImageSets", "Main",
                             "%s.txt" % self.split) for year in [2007, 2012]
            ]
            self.ids = VOCDataset._read_image_ids(image_sets_file)
        else:
            image_sets_file = [
                os.path.join(self.data_dir, f'VOC{year}', "ImageSets", "Main",
                             "%s.txt" % self.split) for year in [2007]
            ]
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
        target = np.zeros((len(boxes), 6))
        for i, (box, label) in enumerate(zip(boxes, labels)):
            target[i, 0] = 0
            target[i, 1] = label / 1.
            w = box[2] - box[0]
            h = box[3] - box[1]
            target[i, 2] = (box[0] + box[2]) / 2
            target[i, 3] = (box[1] + box[3]) / 2
            target[i, 4] = w
            target[i, 5] = h
        target = torch.from_numpy(target)
        return image, target

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_files):
        ids = []
        for filename in image_sets_files:
            sep = '\\' if '\\' in filename else '/'
            with open(filename) as f:
                lst = filename.split(sep)
                lst = lst[:-1]
                lst[2] = 'Annotations'
                for line in f:
                    lst[3] = f'{line.strip()}.xml'
                    ids.append(sep.join(lst))
        return ids

    def _get_annotation(self, image_id):
        annotation_file = image_id
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
        lst = image_id.split('\\')
        lst[2] = 'JPEGImages'
        lst[3] = lst[3].replace('.xml', '.jpg')
        image_file = '\\'.join(lst)
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


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = VOCDataset(data_dir='datasets', split='train')
    load = DataLoader(dataset,
                      batch_size=2,
                      collate_fn=dataset.collate_fn,
                      shuffle=True)
    i = 0
    for ele in load:
        i += 1
        if i > 100:
            break
        print(ele[0].shape)
        print(ele[1].shape)
