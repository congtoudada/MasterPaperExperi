import random
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class VideoTrainTransform:
    def __init__(self, size, scale=(0.35, 1.0), ratio=(1.5, 2.0), random_thresh=0.25, interpolation=InterpolationMode.BICUBIC):
        if not isinstance(scale, tuple) or not len(scale) == 2:
            raise ValueError('Scale should be a tuple with two elements.')

        self.size = size
        self.scale = scale
        self.ratio = ratio  # 宽/高！！！
        self.random_thresh = random_thresh
        self.interpolation = interpolation
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img1, img2, grad):
        # https://blog.csdn.net/qq_36915686/article/details/122136299
        i, j, h, w = transforms.RandomResizedCrop.get_params(img1, self.scale, self.ratio)
        img1 = F.resized_crop(img1, i, j, h, w, self.size, self.interpolation)
        img2 = F.resized_crop(img2, i, j, h, w, self.size, self.interpolation)
        grad = F.resized_crop(grad, i, j, h, w, self.size, self.interpolation)

        if random.random() < self.random_thresh:
            img1 = F.hflip(img1)
            img2 = F.hflip(img2)
            grad = F.hflip(grad)

        # img1.show()
        # grad.show()
        img1 = self.normalize(self.to_tensor(img1))
        img2 = self.normalize(self.to_tensor(img2))
        grad = self.to_tensor(grad)

        return img1, img2, grad


class VideoTestTransform:
    def __init__(self, size):

        self.size = size
        # self.scale = scale
        # self.ratio = (size[0] / size[1], size[0] / size[1])
        # self.interpolation = interpolation
        self.resize = transforms.Resize(self.size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img1, grad):
        # https://blog.csdn.net/qq_36915686/article/details/122136299
        # i, j, h, w = transforms.RandomResizedCrop.get_params(img1, (1.0, 1.0), )
        # img1 = F.resized_crop(img1, i, j, h, w, self.size, self.interpolation)
        # grad = F.resized_crop(grad, i, j, h, w, self.size, self.interpolation)
        # 统一缩放
        img1 = self.resize(img1)
        grad = self.resize(grad)

        # img1.show()
        # grad.show()

        img1 = self.normalize(self.to_tensor(img1))
        grad = self.to_tensor(grad)

        return img1, grad
