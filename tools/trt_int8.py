#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import shutil
from loguru import logger

import tensorrt as trt
import torch
from torch2trt import torch2trt

from yolox.exp import get_exp

import cv2
import numpy as np

from yolox.data.data_augment import ValTransform

class CalibDatasetClass:

    def __init__(self, root, image_size=(640, 640)):
        # Nacteni cest k obrazkum
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.preproc = ValTransform(legacy=False)
        # Normalizacni hodnoty
        self.test_size=image_size

        # Nacteni datasetu do pameti
        self.images = [self._load_image(path) for path in self.image_paths]


    def _load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Obrázek {path} nelze načíst.")
        
        image, _ = self.preproc(image, None, self.test_size)
        image_tensor = torch.from_numpy(image).unsqueeze(0)
        image_tensor = image_tensor.float()
        image_tensor = image_tensor.cuda()

        return image_tensor
    
    def __len__(self):
        """Vrátí počet obrázků v datasetu."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Vrátí obrázek na zadaném indexu s batch dimenzí.
        :param idx: Index obrázku.
        :return: Tensor obrázku ve tvaru vhodnem pro síť yolox
        """
        image = self.images[idx]
        return [image.unsqueeze(0)]  # Přidá batch dimenzi    

def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "-w", '--workspace', type=int, default=32, help='max workspace size in detect'
    )
    parser.add_argument("-d", "--dataset-path", type=str, help='calib dataset path')
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    return parser


@logger.catch
@torch.no_grad()
def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    model.eval()
    model.cuda()
    model.head.decode_in_inference = False
    dataset = CalibDatasetClass(args.dataset_path, exp.test_size)
    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
    model_trt = torch2trt(
        model,
        [x],
        int8_mode=True,
        int8_calib_dataset=dataset,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << args.workspace),
        max_batch_size=args.batch,
    )
    torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")
    engine_file_demo = os.path.join("demo", "TensorRT", "cpp", "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize()) 

    shutil.copyfile(engine_file, engine_file_demo)
    
    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
