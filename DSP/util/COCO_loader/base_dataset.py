from abc import abstractmethod
import torch.utils.data as data
import numpy as np
from pycocotools.coco import COCO
import os
import logging
import cv2
import torch


__all__ = ['BaseDataset']


class BaseDataset(data.Dataset):

    def __init__(self, root, split, cfg, mode=None, base_size=None,
                 crop_size=None, ann_path=None, ann_file_format=None,
                 has_inst_seg=True, **kwargs):
        self.root = root
        self._split = split
        self.mode = mode
        self.base_size = base_size if base_size is not None else 1024
        self.crop_size = crop_size if crop_size is not None else [512, 512]


        self.cfg = cfg


        if split == 'test':
            return

        if ann_path is None:
            ann_path = 'gtFine/annotations_coco_format_v1'
        if ann_file_format is None:
            ann_file_format = 'instances_%s.json'
        self.coco = COCO(os.path.join(root, ann_path, ann_file_format % split))

        # Image paths is currently none to address test split length..
        # update image paths after initializing BaseDataset
        self.image_paths = None
        self.image_ids = list(self.coco.imgs.keys())
        self.image_ids = sorted(self.image_ids)
        logging.info(f'Number of images in split {split} is {len(self.image_ids)}')

        ids = []
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco.loadAnns(ann_ids)
            if split == "train":
                if self.has_valid_annotation(anno):
                    ids.append(img_id)
            else:
                ids.append(img_id)

        self.image_ids = ids
        logging.info(f'Number of images with valid annotations '
                     f'in split {split} is {len(self.image_ids)}')

        self.id_to_filename = dict()
        self.filename_to_id = dict()
        for i, ob in self.coco.imgs.items():
            self.filename_to_id[ob['file_name']] = ob['id']
            self.id_to_filename[ob['id']] = ob['file_name']

        detect_ids = self.get_detect_ids()
        self.coco_id_to_contiguous_id = {coco_id: i for i, coco_id
                                         in enumerate(detect_ids)}
        self.contiguous_id_to_coco_id = {v: k for k, v in
                                         self.coco_id_to_contiguous_id.items()}

        self.key, self.segment_mapping = self.get_segment_mapping()

        self.has_inst_seg = has_inst_seg
        self.inst_encoding_type ='MEINST'



    @property
    def image_size(self):
        return self.crop_size

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, value):
        assert type(value) is str, 'Dataset split should be string'
        self._split = value

    @abstractmethod
    def get_detect_ids(self):
        pass

    @abstractmethod
    def get_segment_mapping(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    def __len__(self):
        if self.split == "test":
            return len(self.image_paths)
        return len(self.image_ids)

    @staticmethod
    def xywh2xyxy(box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def get_img_info(self, index):
        image_id = self.image_ids[index]
        img_data = self.coco.imgs[image_id]
        return img_data



    @abstractmethod
    def ann_check_hooks(self, ann_obj):
        pass

    def get_annotation(self, index):
        image_id = self.image_ids[index]
        # TODO: optionally create segmentation masks...
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        loaded_anns = self.coco.loadAnns(ann_ids)

        bboxes, labels, inst_masks = [], [], []
        for obj in loaded_anns:
            if obj.get('iscrowd', 0) == 0 and obj.get('real_box', True)\
                    and self.ann_check_hooks(obj):
                bboxes.append(self.xywh2xyxy(obj["bbox"]))
                labels.append(self.coco_id_to_contiguous_id[obj["category_id"]])
                if self.has_inst_seg:
                    inst_masks.append(self.coco.annToMask(obj))

        bboxes = np.array(bboxes, np.float32).reshape((-1, 4))
        labels = np.array(labels, np.int64).reshape((-1,))

        # remove invalid boxes
        keep = (bboxes[:, 3] > bboxes[:, 1]) & (bboxes[:, 2] > bboxes[:, 0])
        bboxes = bboxes[keep]
        labels = labels[keep]
        inst_masks = [inst_masks[idx] for idx, k in enumerate(keep) if k]

        rets = [bboxes, labels]
        if self.has_inst_seg:
            rets += [inst_masks]
        return rets

    @staticmethod
    def _has_only_empty_bbox(anno):
        return all(not (obj.get("iscrowd", 0) == 0 and
                        obj.get("real_bbox", True)) for obj in anno)

    def has_valid_annotation(self, anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if self._has_only_empty_bbox(anno):
            return False
        return True

    def add_area(self):
        for i, v in self.coco.anns.items():
            v['area'] = v['bbox'][2] * v['bbox'][3]

    def segment_mask_transform(self, mask):
        mask = np.array(mask).astype('int32')
        if self.segment_mapping is not None:
            mask = self.segment_mask_to_contiguous(mask)
        return torch.from_numpy(mask).long()

    def segment_mask_to_contiguous(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert (values[i] in self.segment_mapping)
        index = np.digitize(mask.ravel(), self.segment_mapping, right=True)
        return self.key[index].reshape(mask.shape)


