# adapted from https://github.com/facebookresearch/moco-v3

from PIL import Image, ImageFilter, ImageOps, ImageFile
import math
import random
import torchvision.transforms.functional as tf
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ThreeCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2, orig_transform):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2
        self.orig_transform = orig_transform

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        im_orig = self.orig_transform(x)
        return [im1, im2, im_orig]
    
class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
