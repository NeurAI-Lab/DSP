from PIL import ImageFilter, Image
import random
from torchvision.transforms import transforms
import numpy as np
import cv2
import skimage.exposure
from scipy.ndimage import gaussian_filter
import pickle
from config.option import Options
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
args = Options().parse()

from util.transforms import RandomChoice

class SimCLRTransform():
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    ]
    """

    def __init__(self, size):
        # Normalize val dataset CMU
        # mean_val: TO= [0.34966046 0.33492374 0.3141161 ] T1= [0.27263916 0.27427372 0.26884845]
        # std_val : T0= [0.3798822  0.37294477 0.35809073] T1= [0.26939082 0.28229916 0.28446007]
        self.T0_mean = (0.33701816, 0.33383232, 0.3245374)
        self.T0_std = (0.26748696, 0.2733889,  0.27516264)
        self.T1_mean = (0.3782613,  0.36675948, 0.35721096)
        self.T1_std = (0.26745927, 0.2732622,  0.2772976)
        self.size = size
        self.copy_paste = args.copy_paste
        if self.size == 512 or self.size == 256:  # CMU
            normalize = transforms.Normalize(mean=self.T0_mean, std=self.T0_std)
            self.train_transform = transforms.Compose(
                [    #transforms.RandomResizedCrop(size=size),
                     transforms.Resize(size=(self.size,self.size)),
                    ])

            self.copy_paste_aug = copy_paste(sigma=3, affine=False, prob=0.5)
            self.train_transform2 = RandomChoice([ get_color_distortion(),
                                                  transforms.RandomApply([GaussianBlur([.1, 2.])], p=1)
                                                                                                    ])

            self.train_transform3 = transforms.Compose([
                                                transforms.ToTensor(),
                                                # transforms.RandomErasing(p=0.4, scale=(0.09, 0.25), ratio=(0.3, 3.3))
                                                normalize,
                                                # transforms.RandomErasing(p=0.5, scale=(0.09, 0.25), ratio=(0.3, 3.3))
                                                # hide_patch(0.2),
                                            ])
            self.train_transform_pcd = transforms.Compose([
                transforms.ToTensor()

            ])
            self.train_transform4 = RandomChoice([
                transforms.RandomErasing(p=0.5, scale=(0.09, 0.25), ratio=(0.3, 3.3))
                # hide_patch(1),
            ])



            self.test_transform = transforms.Compose(
                [
                    transforms.Resize(size=(size, size)),
                    transforms.ToTensor(),
                    normalize
                ]
            )

            self.sup_transform = transforms.Compose(
                [
                    transforms.RandomCrop(size=(size, size)),  # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
            )

    def __call__(self, x1, x2):
        aug1 = self.train_transform(x1)
        aug2 = self.train_transform(x2)
        # if self.copy_paste:
        #     aug1, aug2 = self.copy_paste_aug(aug1, aug2)
        if args.ssl_dataset=='CMU':
            aug1, aug2 = self.train_transform2([aug1, aug2])
            # aug2 = self.train_transform2(aug2)
            # aug1 = self.train_transform2(aug1)
            aug1 = self.train_transform3(aug1)
            aug2 = self.train_transform3(aug2)
        else:
            aug1, aug2 = self.train_transform2([aug1, aug2])



        return aug1, aug2


class GaussianBlur(object):
    """Gaussian blur augmentation """

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class hide_patch(object):
    """" Hide random part of the image """

    def __init__(self, hide_prob=0.3):
        self.hide_prob = hide_prob
        self.skipsize = 20

    def __call__(self, img):
        s = img.shape
        wd = s[1]
        ht = s[2]

        # possible grid size, 0 means no hiding
        if wd ==224:
            grid_sizes = [15, 20, 25]
        else :
            grid_sizes = [33, 44, 55]

        # hiding probability

        # randomly choose one grid size
        grid_size = grid_sizes[random.randint(0, len(grid_sizes) - 1)]

        # hide the patches
        if grid_size != 0:
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if x <= self.skipsize:
                        img[:, x:x_end, y:y_end] = 0

                    if random.random() <= self.hide_prob:
                        # patch_avg = img[:, x:x_end, y:y_end].mean()  # activate this line if u want mean patch value
                        img[:, x:x_end, y:y_end] = 0      # patch_avg

        return img





def get_color_distortion(s=1.0):
    """
    Color jitter from SimCLR paper
    @param s: is the strength of color distortion.
    """

    color_jitter = transforms.ColorJitter(0.6*s, 0.6*s, 0.6*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.7)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class copy_paste(object):
    ''' Copy paste augumentation: arg: paste img, paste mask, img on which the obj to be pasted, gaussian blur(sigma)
        params: sigma = Gaussian blur radius
                blend = bool
                affine =  bool
                instance_txt_path = path to the directory containing the instances list that needs to be pasted.

    '''


    def __init__(self, blend=True, sigma= 1, affine=True, prob=1):
        self.sigma = sigma
        self.blend = blend
        self.affine = affine
        self.prob = prob
        self.instance_txt_path = '/data/input/datasets/VL-CMU-CD/instance.txt'
        with open(self.instance_txt_path, 'rb') as fp:
            self.instance_list = pickle.load(fp)


    def __call__(self, copy_img, copy_img2):
        if random.random() <= self.prob:
            if self.instance_list:
                inst_name = random.choice(self.instance_list)
            instance = Image.open(inst_name)
            self.instance = instance

            if self.instance is not None:
                H,W = copy_img.size
                paste_img = transforms.Resize(size=(H, W))(self.instance)
                if self.affine == True:
                    paste_img = transforms.RandomAffine(degrees=0, translate=(0.25, 0.25), scale=(0.8, 1.1), shear=0)(paste_img)
                gray_mask = transforms.Grayscale()(paste_img)
                binary_mask = np.asarray(gray_mask)
                binary_mask = 1.0 * (binary_mask > 0)
                # blur_binary_mask = skimage.exposure.rescale_intensity(blur_binary_mask)
                invert_mask = 1.0 * (np.logical_not(binary_mask).astype(int))

                if self.blend == True:
                    blur_invert_mask = gaussian_filter(invert_mask, sigma=self.sigma)
                    blur_binary_mask = gaussian_filter(binary_mask, sigma=self.sigma)
                    blur_invert_mask = np.expand_dims(blur_invert_mask, 2)   # Expanding dims to match channels
                    blur_binary_mask = np.expand_dims(blur_binary_mask, 2)
                blur_invert_mask = np.expand_dims(invert_mask, 2)  # Expanding dims to match channels
                blur_binary_mask = np.expand_dims(binary_mask, 2)
                aug_image1 = (paste_img * blur_binary_mask) + (copy_img * blur_invert_mask)
                aug_image2 = (paste_img * blur_binary_mask) + (copy_img2 * blur_invert_mask)


            return Image.fromarray(np.uint8(aug_image1)), Image.fromarray(np.uint8(aug_image2))
        else:
            return(copy_img), (copy_img2)












