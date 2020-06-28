import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib.ticker import NullLocator
from data.dataset import VOCDataset


def test():
    dataset = VOCDataset('datasets', split='train')
    loader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=True,
                        collate_fn=dataset.collate_fn)
    for i, (img, target) in enumerate(loader):
        img = img.permute((0, 2, 3, 1))
        img = img[0]
        mean = torch.Tensor([123, 117, 104])
        #img = img + mean
        arr = img.numpy()*255
        im = Image.fromarray(arr.astype(np.uint8))
        im.save('test.png')
        h, w = im.size
        plt.figure(8)
        plt.imshow(img)
        currentAxis = plt.gca()
        s = torch.Tensor([w, h, w, h])
        for ele in target:
            box = ele[2:]
            box[0] = box[0] - box[2] / 2
            box[1] = box[1] - box[3] / 2
            x = box.unsqueeze(0) * s
            x = x.squeeze(0).numpy()
            x = [int(ele) for ele in x]
            x1, y1 = x[:2]
            box_w, box_h = x[2:]
            bbox = patches.Rectangle((x1, y1),
                                     box_w,
                                     box_h,
                                     linewidth=2,
                                     edgecolor='r',
                                     facecolor="none")
            # Add the bbox to the plot
            currentAxis.add_patch(bbox)
        #plt.gca().xaxis.set_major_locator(NullLocator())
        plt.show()
        #plt.axis("off")
        #plt.savefig(f"output/test.png", bbox_inches="tight", pad_inches=0.0)


test()
