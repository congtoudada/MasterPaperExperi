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


class VadTrainDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        if args.dataset == "avenue":
            data_path = args.avenue_path
        elif args.dataset == "shanghai":
            data_path = args.shanghai_path
        else:
            raise Exception("Unknown dataset!")
        self.transform = VideoTransform(size=args.input_size)
        self.data, self.gradients = self._read_data(data_path)

    def _read_data(self, data_path):
        data = []
        gradients = []
        extension = None
        for ext in IMG_EXTENSIONS:
            if len(list(glob.glob(os.path.join(data_path, "train/frames", f"*/*{ext}")))) > 0:
                extension = ext
                break

        dirs = list(glob.glob(os.path.join(data_path, "train", "frames", "*")))
        for dir in dirs:
            imgs_path = list(glob.glob(os.path.join(dir, f"*{extension}")))
            if platform.system() == "Windows":
                for i in range(len(imgs_path)):
                    imgs_path[i] = imgs_path[i].replace("\\", "/")
            data += imgs_path
            video_name = os.path.basename(dir)
            gradients_path = []
            for img_path in imgs_path:
                gradients_path.append(os.path.join(data_path, "train", "gradients2", video_name,
                                                   f"{int(os.path.basename(img_path).split('.')[0])}.png")
                                      .replace("\\", "/"))
            gradients += gradients_path
        # data: 原始训练图像
        # gradients：运动梯度
        return data, gradients

    def __getitem__(self, index):
        dir_path, frame_no, len_frame_no = self.extract_meta_info(self.data, index)
        img = cv2.imread(self.data[index])
        next_img = self.read_prev_next_frame_if_exists(dir_path, frame_no, direction=1, length=len_frame_no)
        gradient = cv2.imread(self.gradients[index])

        if img.shape[:2] != self.args.input_size or gradient.shape[:2] != self.args.input_size:
            img = cv2.resize(img, self.args.input_size[::-1])
            next_img = cv2.resize(next_img, self.args.input_size[::-1])
            gradient = cv2.resize(gradient, self.args.input_size[::-1])

        img = img.astype(np.float32)
        next_img = next_img.astype(np.float32)
        gradient = gradient.astype(np.float32)

        img = (img - 127.5) / 127.5
        img = np.swapaxes(img, 0, -1).swapaxes(1, -1)
        next_img = (next_img - 127.5) / 127.5
        next_img = np.swapaxes(next_img, 0, -1).swapaxes(1, -1)
        gradient = np.swapaxes(gradient, 0, 1).swapaxes(0, -1)
        return img, next_img, gradient

    def extract_meta_info(self, data, index):
        frame_no = int(data[index].split("/")[-1].split('.')[0])
        dir_path = "/".join(data[index].split("/")[:-1])
        len_frame_no = len(data[index].split("/")[-1].split('.')[0])
        return dir_path, frame_no, len_frame_no

    def read_prev_next_frame_if_exists(self, dir_path, frame_no, direction=-3, length=1):
        frame_path = dir_path + "/" + str(frame_no + direction).zfill(length) + ".png"
        if os.path.exists(frame_path):
            return cv2.imread(frame_path)
        else:
            return cv2.imread(dir_path + "/" + str(frame_no).zfill(length) + ".png")

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
    vtd = VadTrainDataset(args)
    print(f"len: {vtd.__len__()}")  # avenue: 15328
    pkg = vtd.__getitem__(vtd.__len__() - 1)
    print(f"item: {pkg[0].shape} {pkg[1].shape} {pkg[2].shape}")
