from collections import Counter
from os.path import join as pjoin, splitext as spt
import os
import cv2
from shutil import copyfile
from pathlib import Path
dict = {'1_00':0,'1_01':1,'1_02':2,'1_03':3,'1_04':4,'1_05':5,'1_06':6,'1_07':7,'1_08':8,'1_09':9,'1_10':10,'1_11':11,'1_12':12,'1_13':13,'1_14':14,'1_15':15,'1_16':16,'1_17':17,'1_18':18,'1_19':19}
pt = '/data/input/datasets/VL-CMU-CD/struc_train/'
path_txt = '/data/input/datasets/VL-CMU-CD/struc_train/train_50p_cmu.txt'
label = '/data/input/datasets/VL-CMU-CD/vl_cmu_cd_binary_mask/vl_cmu_cd_binary_mask/train/mask_struc'
# path = '/data/input/datasets/VL-CMU-CD/struc_train/train_split.txt'
lst = []
count = 0
count_dict = {}
full_list = []
count = 0
datapath = '/data/input/datasets/VL-CMU-CD/struc_train'
labpath = '/data/input/datasets/VL-CMU-CD/vl_cmu_cd_binary_mask/vl_cmu_cd_binary_mask/train/mask_struc'
sav = '/data/input/datasets/VL-CMU-CD/vl_cmu_cd_binary_mask/vl_cmu_cd_binary_mask/dtranet_vl_50pdata'
nameing = 1
for idx, did in enumerate(open(path_txt)):
    try:
        image1_name, image2_name, mask_name = did.strip("\n").split(' ')
    except ValueError:  # Adhoc for test.
        image_name = mask_name = did.strip("\n")
    extract_name = image1_name[image1_name.rindex('/') + 1: image1_name.rindex('.')]

    folder = image1_name.split('/')
    img1_file = os.path.join(pt, image1_name)
    img2_file = os.path.join(pt, image2_name)
    # items = len([name for name in os.listdir(fol_path)])
    imgno = os.path.splitext(folder[2])[0]
    lbl_file = os.path.join(labpath, folder[1])
    filename = list(spt(f)[0] for f in os.listdir(lbl_file))
    filename.sort()
    lbl_file2 = os.path.join(lbl_file, filename[dict[imgno]]+'.png')

    print(img1_file)
    img_t0 = cv2.imread(img1_file, 1)
    img_t1 = cv2.imread(img2_file, 1)
    mask = cv2.imread(lbl_file2, 0)
    #rotate image
    image_t0_90 = cv2.rotate(img_t0, cv2.cv2.ROTATE_90_CLOCKWISE)
    image_t0_180 = cv2.rotate(image_t0_90, cv2.cv2.ROTATE_90_CLOCKWISE)
    image_t0_270 = cv2.rotate(image_t0_180, cv2.cv2.ROTATE_90_CLOCKWISE)
    image_t1_90 = cv2.rotate(img_t1, cv2.cv2.ROTATE_90_CLOCKWISE)
    image_t1_180 = cv2.rotate(image_t1_90, cv2.cv2.ROTATE_90_CLOCKWISE)
    image_t1_270 = cv2.rotate(image_t1_180, cv2.cv2.ROTATE_90_CLOCKWISE)
    mask_90 = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)
    mask_180 = cv2.rotate(mask_90, cv2.cv2.ROTATE_90_CLOCKWISE)
    mask_270 = cv2.rotate(mask_180, cv2.cv2.ROTATE_90_CLOCKWISE)
    print(pjoin(sav,'t0',str(nameing)+'.png'))
    cv2.imwrite(pjoin(sav,'t0',str(nameing)+'.png'), img_t0)
    cv2.imwrite(pjoin(sav,'t0',str(nameing+1)+'.png'), image_t0_90)
    cv2.imwrite(pjoin(sav,'t0',str(nameing+2)+'.png'), image_t0_180)
    cv2.imwrite(pjoin(sav,'t0',str(nameing+3)+'.png'), image_t0_270)
    cv2.imwrite(pjoin(sav, 't1', str(nameing) + '.png'), img_t1)
    cv2.imwrite(pjoin(sav, 't1', str(nameing + 1) + '.png'), image_t1_90)
    cv2.imwrite(pjoin(sav, 't1', str(nameing + 2) + '.png'), image_t1_180)
    cv2.imwrite(pjoin(sav, 't1', str(nameing + 3) + '.png'), image_t1_270)
    cv2.imwrite(pjoin(sav, 'mask', str(nameing) + '.png'), mask)
    cv2.imwrite(pjoin(sav, 'mask', str(nameing + 1) + '.png'), mask_90)
    cv2.imwrite(pjoin(sav, 'mask', str(nameing + 2) + '.png'), mask_180)
    cv2.imwrite(pjoin(sav, 'mask', str(nameing + 3) + '.png'), mask_270)

    nameing = nameing+4
# with open(path) as g:
#     for line in g:
#         ls= line.split()
# datapath = '/data/input/datasets/VL-CMU-CD/vl_cmu_cd_binary_mask/vl_cmu_cd_binary_mask/train/mask_900images'
# formatpath = '/data/input/datasets/VL-CMU-CD/struc_train/gt_fold_rgb'
# filename = list(spt(f)[0] for f in os.listdir(datapath) )
# filename.sort()
# print(filename)
# query_item = 0
# for word in ls :
#     word1 = word.zfill(3)
#     fol_path = pjoin(formatpath, word1)
#     items = len([name for name in os.listdir(fol_path)])
#     savepath = '/data/input/datasets/VL-CMU-CD/vl_cmu_cd_binary_mask/vl_cmu_cd_binary_mask/train/mask_struc'
#     Path(os.path.join(savepath, word1)).mkdir(parents=True, exist_ok=True)
#     for id in range(items):
#         q = query_item +id
#         copyfile(pjoin(datapath,filename[q]+'.png'), pjoin(savepath,word1,filename[q]+'.png'))
#
#     query_item = items + query_item
