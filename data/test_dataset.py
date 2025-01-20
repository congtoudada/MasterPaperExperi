import argparse
import glob
import os
import platform

import cv2
import numpy as np
import torch.utils.data

from configs.configs import get_configs_avenue, get_configs_shanghai
from data.mae_dataset import VideoTransform

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]


class VadTestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        if args.dataset == "avenue":
            data_path = args.avenue_path
            gt_path = args.avenue_gt_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
            gt_path = args.shanghai_gt_path
        else:
            raise Exception("Unknown dataset!")
        self.transform = VideoTransform(size=args.input_size)
        self.data, self.labels, self.gradients = self._read_data(data_path, gt_path)

    def _read_data(self, data_path, gt_path):
        data = []
        labels = []
        gradients = []

        extension = None
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "test/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break
        self.extension = extension
        dirs = list(glob.glob(os.path.join(data_path, "test", "frames", "*")))
        for dir in dirs:
            imgs_path = list(glob.glob(os.path.join(dir, f"*{extension}")))
            if platform.system() == "Windows":
                for i in range(len(imgs_path)):
                    imgs_path[i] = imgs_path[i].replace("\\", "/")
            imgs_path = sorted(imgs_path, key=lambda x: int(os.path.basename(x).split('.')[0]))
            lbls = np.loadtxt(os.path.join(gt_path, f"{os.path.basename(dir)}.txt"))

            data += imgs_path
            labels += list(lbls)

            video_name = os.path.basename(dir)
            gradients_path = list(glob.glob(os.path.join(data_path, "test", "gradients2", video_name, "*.png")))
            gradients_path = sorted(gradients_path, key=lambda x: int(os.path.basename(x).split('.')[0]))
            gradients += gradients_path
        # data: 原始训练图像
        # gradients：运动梯度
        return data, labels, gradients

    def __getitem__(self, index):
        img = cv2.imread(self.data[index])
        gradient = cv2.imread(self.gradients[index])
        if img.shape[:2] != self.args.input_size[::-1]:
            img = cv2.resize(img, self.args.input_size[::-1])
            gradient = cv2.resize(gradient, self.args.input_size[::-1])

        img = img.astype(np.float32)
        gradient = gradient.astype(np.float32)

        img = (img - 127.5) / 127.5
        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        gradient = np.swapaxes(gradient, 0, 1).swapaxes(0, -1)
        return img, gradient, self.labels[index], self.data[index].split('/')[-2], self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.__class__.__name__


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avenue')
    args = parser.parse_args()
    if args.dataset == 'avenue':
        args = get_configs_avenue()
    else:
        args = get_configs_shanghai()
    vtd = VadTestDataset(args)
    print(f"len: {vtd.__len__()}")  # avenue: 15324
    pkg = vtd.__getitem__(vtd.__len__() - 1)
    print(f"item: {pkg[0].shape} {pkg[1].shape} {pkg[2].shape}")