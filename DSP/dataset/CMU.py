import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from config.option import Options
import matplotlib.pyplot as plt

args = Options().parse()


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    print(filename)
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


palette = [0, 0, 0,255,0,0]

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_pascal_labels():
    return np.asarray([[0,0,0],[0,0,255]])

def decode_segmap(temp, plot=False):

    label_colours = get_pascal_labels()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 2):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    #rgb = np.resize(rgb,(321,321,3))
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


class Dataset(Dataset):

    def __init__(self,data_path,split_flag, flag_type= 'ssl', transform=False, transform_med=None):
        self.size = args.img_size
        self.train_data_path = os.path.join(data_path, "struc_train")
        self.test_data_path = os.path.join(data_path, 'struc_test')
        self.img_txt_path =  os.path.join(self.train_data_path, 'train_pair.txt')
        self.test_img_txt_path =  os.path.join(self.test_data_path, 'test_pair.txt')
        print( self.test_img_txt_path)
        # # Load the text file containing image pair
        # self.imgs_path_list = np.loadtxt(self.img_txt_path,dtype=str)
        # self.test_imgs_path_list = np.loadtxt(self.test_img_txt_path,dtype=str)

        self.flag = split_flag
        self.flag_type = flag_type
        self.transform = transform
        self.transform_med = transform_med
        self.img_label_path_pairs = self.get_img_label_path_pairs()

    def get_img_label_path_pairs(self):

        img_label_pair_list = {}
        if self.flag =='train':
            for idx, did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name,image2_name,mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')]
                img1_file = os.path.join(self.train_data_path, image1_name)
                img2_file = os.path.join(self.train_data_path, image2_name)
                lbl_file = os.path.join(self.train_data_path, mask_name)
                img_label_pair_list.setdefault(idx, [img1_file,img2_file,lbl_file, image1_name, image2_name])

        if self.flag == 'val':
            self.label_ext = '.png'
            for idx , did in enumerate(open(self.test_img_txt_path)):
                try:
                    image1_name, image2_name, mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                # extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')]
                img1_file = os.path.join(self.test_data_path, image1_name)
                img2_file = os.path.join(self.test_data_path, image2_name)
                lbl_file = os.path.join(self.test_data_path, mask_name)
                img_label_pair_list.setdefault(idx, [img1_file, img2_file, lbl_file, image1_name, image2_name])

        return img_label_pair_list

    def data_transform(self, img1,img2,lbl):
       rz = transforms.Compose([transforms.Resize(size=(512,512))])
       img1 = rz(img1)
       img2 = rz(img2)
       lbl= transforms.ToPILImage()(lbl)
       lbl = rz(lbl)
       img1 = transforms.ToTensor()(img1)
       img2 = transforms.ToTensor()(img2)
       lbl = transforms.ToTensor()(lbl)
        #lbl_reverse = torch.from_numpy(lbl_reverse).long()
       return img1,img2,lbl

    def extract_instance(self, img1, img2, lbl):

        obj_mask = 1*(lbl >1)
        img1 = np.array(img1)
        gau_masks = gaussian_filter(obj_mask, sigma=1)
        gau_masks = np.reshape(gau_masks, (gau_masks.shape[0], gau_masks.shape[1], 1))
        instance = img1* gau_masks
        transform = transforms.ToTensor()
        img1 = transform(img1)
        img2 = transform(img2)
        obj_mask = transform(obj_mask)
        instance = transform(instance)
        return img1, img2, obj_mask, instance

    def __getitem__(self, index):

        img1_path,img2_path,label_path,filename1, filename2 = self.img_label_path_pairs[index]
        # print(img1_path,filename1,filename2)
        ####### load images #############
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        label = Image.open(label_path)
        label = np.array(label, dtype=np.int32)

        height,width, d = np.array(img1,dtype= np.uint8).shape

        if self.transform_med != None:
            # normal simclr
           img1_0, img2_0  = self.transform_med(img1, img2)
           img1_1, img2_1= self.transform_med(img1,img2)

        if self.flag_type == 'ssl':
            return img1_0, img1_1, img2_0, img2_1, str(filename1), str(filename2), label
        elif self.flag_type == 'linear_eval':
               image_dict = {'pos1': img1_0, 'pos2': img1_1, 'neg1': img2_0, 'neg2': img2_1}
               type1, type2 = random.sample(list(image_dict.keys()), k=2)

               if any('pos' in s for s in [type1, type2]) and any('neg' in s for s in [type1, type2]):
                   y = 1
                   i1 = image_dict[type1]
                   i2 = image_dict[type2]
                   return i1, i2, y
               else:
                   y = 0
                   i1 = image_dict[type1]
                   i2 = image_dict[type2]
                   return i1, i2, y

        ####### load labels ############
        if self.flag == 'train':
            label = Image.open(label_path)
            # if self.transform_med != None:                # enable this during fine tuning
            #     label = self.transform_med(label)
            label = np.array(label,dtype=np.int32)

        if self.flag == 'val':
            label = Image.open(label_path)
            # if self.transform_med != None:                # enable this during fine tuning
            #    label = self.transform_med(label)
            label = np.array(label,dtype=np.int32)

        if self.transform :
            img1, img2, label = self.data_transform(img1,img2,label) #self.extract_instance(img1, img2, label)

            return img1, img2, label


        else:
            return img1, img2, label
    def __len__(self):

        return len(self.img_label_path_pairs)



