import os
import numpy as np
from pycocotools import mask as coco_mask
import logging
from PIL import Image
import dataset.CMU as CMU
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import torch
import random
from torch.utils.data import DataLoader



from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from util.COCO_loader.base_dataset import BaseDataset


class COCOUninet(BaseDataset):
    NUM_CLASSES = {'segment': 21, 'detect': 81, 'inst_seg': 81}
    INSTANCE_NAMES = []

    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]

    def __init__(self, root=os.path.expanduser('/data/input/datasets/mscoco'),
                 split='train', mode=None, cfg=None, **kwargs):
        year = str(2017)
        if year == "2017" and split == 'minival':
            split = 'val'
        super(COCOUninet, self).__init__(
            root, split, cfg, mode, ann_path='annotations',
            ann_file_format=f'instances_%s{year}.json', **kwargs)

        if self.split == "test":
            self.image_paths = get_image_paths(self.root, year=year)
            if len(self.image_paths) == 0:
                raise RuntimeError("Found 0 images in subfolders of:" + self.root + "\n")
            return

        self.img_dir = os.path.join(root, rf'{split}{year}')
        self.add_area()

    @staticmethod
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def ann_check_hooks(self, obj):
        return True

    def get_detect_ids(self):
        det_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18,
                   19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37,
                   38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                   55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
                   75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

        return det_ids

    def get_segment_mapping(self):
        key = None
        segment_mapping = None

        return key, segment_mapping

    def __getitem__(self, index):
        if self.split == "test":
            image = Image.open(self.image_paths[index]).convert('RGB')
            image = np.array(image)
            image, _ = self.transform(image)
            return image

        file_name = self.id_to_filename[self.image_ids[index]]
        image_path = os.path.join(self.img_dir, file_name)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        bboxes, labels, inst_masks = self.get_annotation(index)

        inst_list = []
        for idx in range(len(inst_masks)):

            if labels[idx] == 3:
                class_name = labels[idx]
                masks = np.array(inst_masks[idx])
                masks = np.reshape(masks, (masks.shape[0], masks.shape[1], 1))
                gau_masks = gaussian_filter(masks, sigma=1)

                instance = image * gau_masks
                nzCount = instance.any(axis=-1).sum()
                print(nzCount)
                if nzCount > 10000 and nzCount < 60000:
                    inst_list.append(instance)

        if labels[idx] == 3:
            return image, inst_list, gau_masks, class_name
        else:
            return image






def get_image_paths(folder, split='test', year='2014'):
    def get_path_pairs():
        img_paths = []
        for root, directories, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    im_path = os.path.join(root, filename)
                    if os.path.isfile(im_path):
                        img_paths.append(im_path)
                    else:
                        logging.info('cannot find the mask or image:', im_path)
        logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths

    img_folder = os.path.join(folder, split + year)
    return get_path_pairs()

def convert_togray (cd_img1, instance):
    img1 = instance
    b,c,H,W = cd_img1.shape
    size = (H,W)
    transform = transforms.ToPILImage()
    img1  = img1.squeeze(0)
    img1 = img1.permute(2, 0, 1)
    img1 = transform(img1)

    img1 = transforms.Resize(size=size)(img1)
    gray_instance = transforms.Grayscale()(img1)
    gray_instance, resized_instance = transforms.ToTensor()(gray_instance),transforms.ToTensor()(img1)

    return gray_instance.unsqueeze(0), resized_instance.unsqueeze(0)

def image_copy_paste(img1, img2, instance, alpha, blend=True, sigma=1):
    if alpha is not None:
        gray_ins, instance = convert_togray(img1, instance)
        binarized = 1.0 * (gray_ins > 0)
        invert_binary = (~binarized).float()
        if blend:
            filtered_mask = gaussian_filter(invert_binary, sigma=1)
            filtered_mask = torch.Tensor(filtered_mask)
        aug_img1 = instance + (img1 * filtered_mask)
        aug_img2 = instance + (img2 * filtered_mask)

    return aug_img1, aug_img2, instance

def save_show_transformations(img, img2,instance, masks, name, path = '/data/input/datasets/VL-CMU-CD/instances'):

    img = np.squeeze(img)
    img2 = np.squeeze(img2)
    instance = np.squeeze(instance)
    masks = np.squeeze(masks)
    instance = instance.permute(1,2,0)
    img= np.transpose(img,(0,1, 2))   # from NCHW to NHWC
    instance= np.transpose(instance,(0,1, 2))
    f, axarr = plt.subplots(1,4)
    axarr[0].imshow(img.permute(1,2,0))
    axarr[1].imshow(img2.permute(1,2,0))
    axarr[2].imshow(instance)
    axarr[3].imshow(masks)
    plt.show()


    instance = instance.numpy()
    rescaled = (255.0 / instance.max() * (instance - instance.min())).astype(np.uint8)
    Name_Formatted = ("%s" % (j)) + ".png"
    # file_path = os.path.join(path, Name_Formatted)
    # instance = Image.fromarray(rescaled)
    # instance.save(file_path)
    return file_path



# COCO dataset loader
coco_train_dataset = COCOUninet()
train_loader_coco = DataLoader(coco_train_dataset, batch_size=1, shuffle=True, drop_last=True)
# Change detection dataset loader
TRAIN_DATA_PATH = "/data/input/datasets/VL-CMU-CD/struc_train"
data_path = os.path.join(TRAIN_DATA_PATH, 'train_pair.txt')
CD_train_dataset = CMU.Dataset(TRAIN_DATA_PATH, TRAIN_DATA_PATH,
                               data_path, 'train', 'CD', transform=True,
                            transform_med=None)
train_loader_CD = DataLoader(CD_train_dataset, batch_size=1, shuffle=True, drop_last=True)

def extract_ins_coco ():
    for j, batch_CD in enumerate(train_loader_CD):
        t0, t1, cd_labels, instance = batch_CD
        for i, batch in enumerate(train_loader_coco):
            if len(batch) >1:
                img, instance, masks, labels = batch
                if instance:
                    for ins in instance:
                        aug_t0, aug_t1, resized_instance  = image_copy_paste(t0, t1, ins, masks)
                        # show_transformations(img, ins, masks)
                        save_show_transformations(aug_t0, resized_instance, masks)
ins_path = []
for j, batch_CD in enumerate(train_loader_CD):
    t0, t1, cd_labels, ins = batch_CD
    file_path = save_show_transformations(t0, t1, ins, cd_labels, j)
    ins_path.append(file_path)


