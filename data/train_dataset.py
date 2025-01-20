import argparse
import platform
import os
import glob
from torch.utils.data import Dataset
import torch.utils.data
from PIL import Image
from configs.configs import get_configs_avenue, get_configs_shanghai
from data.vid_transform import VideoTrainTransform

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
        self.folder_cnt = 0
        self.transform = VideoTrainTransform(size=args.input_size)
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
            self.folder_cnt += 1
            imgs_path = list(glob.glob(os.path.join(dir, f"*{extension}")))
            if platform.system() == "Windows":
                for i in range(len(imgs_path)):
                    imgs_path[i] = imgs_path[i].replace("\\", "/")
            data += list(zip(imgs_path[:-1], imgs_path[1:]))
            video_name = os.path.basename(dir)
            gradients_path = []
            for img_path in imgs_path:
                gradients_path.append(os.path.join(data_path, "train", "gradients2", video_name,
                                                   f"{int(os.path.basename(img_path).split('.')[0])}.png")
                                      .replace("\\", "/"))
            gradients += gradients_path[:-1]

        return data, gradients

    def __getitem__(self, index):
        frame1_path, frame2_path = self.data[index]
        frame1 = Image.open(frame1_path)
        frame2 = Image.open(frame2_path)
        gradient = Image.open(self.gradients[index])

        return self.transform(frame1, frame2, gradient)

    def __len__(self):
        return len(self.data)  # PS: 比实际数据集少1*self.folder_cnt

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
    pkg = vtd.__getitem__(0)
    print(f"item: {pkg[0].shape} {pkg[1].shape} {pkg[2].shape}")  # 3 320 640
