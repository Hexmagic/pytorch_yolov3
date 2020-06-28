from __future__ import division

from yolov3 import *
from utils.utils import *

# from utils.datasets import *
from utils.parse_config import *
from data.dataset import VOCDataset

from terminaltables import AsciiTable
from utils.lr_scheduler import make_lr_scheduler, make_optimizer
import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="size of each image batch")
    parser.add_argument(
        "--model_def",
        type=str,
        default="config/yolov3.cfg",
        help="path to model definition file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets",
        help="path to data config file",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        help="if specified starts from checkpoint model",
    )
    parser.add_argument("--save_folder",
                        type=str,
                        help="save folder",
                        default="checkpoints")
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument('--iter', type=int, default=120000)
    parser.add_argument("--img_size",
                        type=int,
                        default=416,
                        help="size of each image dimension")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="interval between saving model weights",
    )
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    # Get dataloader
    dataset = VOCDataset(data_dir=opt.data_dir, split="train")
    dataloader = iter(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        ))

    #optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, amsgrad=True)
    optimizer = make_optimizer(model)
    lr_scheduler = make_lr_scheduler(optimizer)
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]
    model.train()
    for iter_i in range(1, opt.iter):
        try:
            imgs, targets = next(dataloader)
        except:
            dataloader = iter(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=opt.batch_size,
                    shuffle=True,
                    num_workers=opt.n_cpu,
                    pin_memory=True,
                    collate_fn=dataset.collate_fn,
                ))
            imgs, targets = next(dataloader)
        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)

        loss, outputs = model(imgs, targets.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        log_str = "\n---- [Iter %d/%d] ----\n" % (
            iter_i,
            opt.iter,
        )

        metric_table = [[
            "Metrics",
            *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]
        ]]

        # Log metrics at each YOLO layer
        for i, metric in enumerate(metrics):
            formats = {m: "%.6f" for m in metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [
                formats[metric] % yolo.metrics.get(metric, 0)
                for yolo in model.yolo_layers
            ]
            metric_table += [[metric, *row_metrics]]

        log_str += AsciiTable(metric_table).table
        log_str += f"\nTotal loss {loss.item()}"

        # Determine approximate time left for epoch
        epoch_batches_left = opt.iter - (iter_i + 1)
        if iter_i % 10 == 0:
            print(log_str)
        model.seen += imgs.size(0)

        if iter_i % 2000 == 0:
            torch.save(model.state_dict(),
                       f"{opt.save_folder}/yolov3_ckpt_%d.pth" % iter_i)
