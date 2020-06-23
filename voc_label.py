import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm
sets = ['train', 'val']
classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(root,image_id):
    in_file = open(f'{root}/Annotations/%s.xml' % (image_id))
    out_file = open(f'{root}/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(
            str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def split_train_val(root):
    dirs = os.listdir(f'{root}/JPEGImages')
    length = len(dirs)
    pos = int(0.8 * length)
    return dirs[:pos], dirs[pos:]


def write_to_main(root,nameList, filename):
    with open(f'{root}/ImageSets/Main/{filename}', 'w') as f:
        for name in nameList:
            f.write(name.replace('.jpg', '') + '\n')


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--root',type=str,default='data')
    param = parser.parse_args()

    root = param.root
    wd = getcwd()
    print(wd)
    train, val = split_train_val(root)
    write_to_main(root,train, 'train.txt')
    write_to_main(root,val, 'val.txt')
    print(f"write Tain {len(train)} Val Text {len(val)}")
    for image_set in sets:
        if not os.path.exists(f'{root}/labels/'):
            os.makedirs(f'{root}/labels/')
        image_ids = open(f'{root}/ImageSets/Main/%s.txt' %
                         (image_set)).read().strip().split()
        list_file = open(f'{root}/%s.txt' % (image_set), 'w')
        for image_id in tqdm(image_ids,desc=image_set):
            list_file.write(f'{root}/JPEGImages/%s.jpg\n' % (image_id))
            convert_annotation(root,image_id)
        list_file.close()


if __name__ == '__main__':
    main()
