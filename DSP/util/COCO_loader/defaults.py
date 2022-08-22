from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# MODEL options
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.PRETRAINED_PATH = "/input/datasets/uninet/pytorch-config/FCOS_imprv_R_50_FPN_1x.pth"
_C.MODEL.IS_FULL_MODEL = False
_C.MODEL.LOAD_BACKBONE = False
_C.MODEL.BACKBONE_NAME = 'backbone'
_C.MODEL.NECK_NAMES = ['fpn', 'neck']
_C.MODEL.HEAD_NAME = 'head'
_C.MODEL.USE_DCN = False

# -----------------------------------------------------------------------------
# INPUT options
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
# ---------------------------------------------------------------------------- #
_C.TEST.BBOX_AUG = CN()
# Enable test-time augmentation for bounding box detection if True
_C.TEST.BBOX_AUG.ENABLED = False

# --------------------------------------------------------------------------- #
# Dataloader Options
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.YEAR = 2014
_C.DATALOADER.ANNOTATION_FOLDER = 'gtFine/annotations_coco_format_v1'
_C.DATALOADER.ANN_FILE_FORMAT = 'instances_%s.json'
# ImageNet mean and standard deviation..
_C.DATALOADER.MEAN = [.485, .456, .406]
_C.DATALOADER.STD = [.229, .224, .225]
_C.DATALOADER.TRAIN_TRANSFORMS = ['PreProcessBoxes', 'PadIfNeeded', 'ShiftScaleRotate', 'CropNonEmptyMaskIfExists',
                                  'ResizeMultiScale', 'HorizontalFlip', 'ColorJitter', 'PostProcessBoxes',
                                  'ConvertFromInts', 'ToTensor', 'Normalize']
_C.DATALOADER.VAL_TRANSFORMS = ['PreProcessBoxes', 'Resize', 'PostProcessBoxes',
                                'ToTensor', 'Normalize']
# Multi scale augmentation defaults..
_C.DATALOADER.MS_MULTISCALE_MODE = 'value'
_C.DATALOADER.MS_RATIO_RANGE = [0.75, 1]
_C.DATALOADER.PHOTOMETRIC_DISTORT_KWARGS = '{}'
_C.DATALOADER.INST_SEG_ENCODING = 'MEINST'
_C.DATALOADER.DEPTH_SCALE = 512.

# ---------------------------------------------------------------------------- #
# Task options
# ---------------------------------------------------------------------------- #
_C.TASKS = CN()
_C.TASKS.TASK_TO_LOSS_NAME = '{\"detect\":"default",\"segment\":"default",\"depth\":"default",' \
                             '\"inst_depth\":"default",\"inst_seg\":"default"}'
_C.TASKS.TASK_TO_LOSS_ARGS = '{}'
_C.TASKS.TASK_TO_LOSS_KWARGS = '{}'
_C.TASKS.TASK_TO_CALL_KWARGS = '{\"segment\":{\"ignore_index\":-1}}'
_C.TASKS.TASK_TO_MIN_OR_MAX = '{\"detect\":1,\"segment\":1,\"depth\":-1,\"inst_depth\":-1,' \
                              ' \"inst_seg\":1}'
_C.TASKS.ALL_LOSSES = ['detect_cls_loss', 'detect_reg_loss', 'detect_centerness_loss',
                       'segment_loss', 'depth_loss', 'inst_depth_l1_loss',
                       'inst_seg_loss']
_C.TASKS.LOSS_INIT_WEIGHTS = [1., 1., 1., 1., 1., 0.05, 1.]
_C.TASKS.LOSS_START_EPOCH = [1, 1, 1, 1, 1, 1, 1]
_C.TASKS.USE_UNCERTAINTY_WEIGHTING = False

# --------------------------------------------------------------------------- #
# Backbone and encoder Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.NUM_EN_FEATURES = 6
_C.MODEL.ENCODER.OUT_CHANNELS_BEFORE_EXPANSION = 512
_C.MODEL.ENCODER.FEAT_CHANNELS = [2048, 2048, 2048, 2048]
_C.MODEL.ENCODER.USE_DCN = False

# --------------------------------------------------------------------------- #
# Decoder Options
# ---------------------------------------------------------------------------- #
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.OUTPLANES = 64
_C.MODEL.DECODER.MULTISCALE = False
_C.MODEL.DECODER.ATTENTION = False
_C.MODEL.DECODER.INSERT_MEAN_FEAT = False
_C.MODEL.DECODER.INIT_WEIGHTS = False
_C.MODEL.DECODER.USE_NECK_FEATURES = False

# --------------------------------------------------------------------------- #
# Object Detection Options
# ---------------------------------------------------------------------------- #
_C.MODEL.DET = CN()
_C.MODEL.DET.HEAD_NAME = "FCOS"
_C.MODEL.DET.FEATURE_CHANNELS = 256
_C.MODEL.DET.FPN_STRIDES = [8, 16, 32]
_C.MODEL.DET.WEIGHTS_PER_CLASS = [1] * 8
_C.MODEL.DET.ATTENTION = False
_C.MODEL.DET.CLS_LOSS_TYPE = 'focal_loss'
# Focal loss parameter: alpha
_C.MODEL.DET.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.DET.LOSS_GAMMA = 2.0
_C.MODEL.DET.LOSS_BETA = 0.9999
_C.MODEL.DET.PRIOR_PROB = 0.01

# --------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.INFERENCE_TH = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOP_N = 1000
# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CONVS = 4
# if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
_C.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 0.0
# IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
_C.MODEL.FCOS.IOU_LOSS_TYPE = "iou"
_C.MODEL.FCOS.NORM_REG_TARGETS = False
_C.MODEL.FCOS.CENTERNESS_ON_REG = False
_C.MODEL.FCOS.USE_DCN_IN_TOWER = False
_C.MODEL.FCOS.USE_NAS_HEAD = False
_C.MODEL.FCOS.ATSS_TOPK = 9

# --------------------------------------------------------------------------- #
# OnetNet Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ONENET = CN()
_C.MODEL.ONENET.CLASS_WEIGHT = 1.
_C.MODEL.ONENET.GIOU_WEIGHT = 1.
_C.MODEL.ONENET.L1_WEIGHT = 2.5
_C.MODEL.ONENET.USE_NMS = False
_C.MODEL.ONENET.NMS_TH = 0.5

# --------------------------------------------------------------------------- #
# Segmentation Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SEG = CN()
_C.MODEL.SEG.INPLANES = 64
_C.MODEL.SEG.OUTPLANES = 64
_C.MODEL.SEG.MULTISCALE = False
_C.MODEL.SEG.ATTENTION = False

# Depth Options
# ---------------------------------------------------------------------------- #
_C.MODEL.DEPTH = CN()
_C.MODEL.DEPTH.INPLANES = 64
_C.MODEL.DEPTH.OUTPLANES = 64
_C.MODEL.DEPTH.ACTIVATION_FN = 'sigmoid'
_C.MODEL.DEPTH.ATTENTION = False

# --------------------------------------------------------------------------- #
# Instance depth Options
# ---------------------------------------------------------------------------- #
_C.MODEL.INST_DEPTH = CN()
_C.MODEL.INST_DEPTH.DEPTH_ON_REG = True

# --------------------------------------------------------------------------- #
# Instance segmentation Options
# ---------------------------------------------------------------------------- #
_C.MODEL.INST_SEG = CN()
_C.MODEL.INST_SEG.HEAD_NAME = 'MEINST'

# --------------------------------------------------------------------------- #
# MEINST Options

_C.MODEL.MEINST = CN()
# share classification head and instance segmentation head..
_C.MODEL.MEINST.SHARE_CLS_INST_HEADS = False
# share bounding box head and instance segmentation head..
_C.MODEL.MEINST.SHARE_BBOX_INST_HEADS = True
# mask encoding type
_C.MODEL.MEINST.ENCODING_TYPE = 'explicit'
# is inverse sigmoid and sigmoid used for finding pca components
_C.MODEL.MEINST.SIGMOID = True
# is whiten used for finding pca components
_C.MODEL.MEINST.WHITEN = True
# path to pca params file
_C.MODEL.MEINST.PCA_PATH = ''
# number of components in the encoded mask
_C.MODEL.MEINST.NUM_COMPONENTS = 60
# dimension to which all instance masks are reshaped to
_C.MODEL.MEINST.ENCODING_DIM = 28
# add instance masks vizualized as segmentation masks to tensorboard
_C.MODEL.MEINST.CREATE_PRED_MASK = False
# visualize each instance separately..
_C.MODEL.MEINST.VIZ_INSTANCES = True


# --------------------------------------------------------------------------- #
# CenterMask Options
# ---------------------------------------------------------------------------- #

_C.MODEL.CENTER_MASK = CN()
_C.MODEL.CENTER_MASK.IN_FEATURES = ['p3', 'p4', 'p5']
_C.MODEL.CENTER_MASK.POOLER_RESOLUTION = 14
_C.MODEL.CENTER_MASK.POOLER_SAMPLING_RATIO = 0
_C.MODEL.CENTER_MASK.POOLER_TYPE = 'ROIAlignV2'
_C.MODEL.CENTER_MASK.ASSIGN_CRITERION = 'ratio'
_C.MODEL.CENTER_MASK.MASK_CONV_DIM = 128
_C.MODEL.CENTER_MASK.MASK_NUM_CONV = 2
_C.MODEL.CENTER_MASK.MASKIOU_CONV_DIM = 128
_C.MODEL.CENTER_MASK.MASKIOU_NUM_CONV = 2
_C.MODEL.CENTER_MASK.CLS_AGNOSTIC_MASK = False
_C.MODEL.CENTER_MASK.MASKIOU_ON = False
_C.MODEL.CENTER_MASK.MASKIOU_LOSS_WEIGHT = 1.0

# --------------------------------------------------------------------------- #
# Miscellaneous Options
# ---------------------------------------------------------------------------- #

_C.MISC = CN()
_C.MISC.CITYS_INST_SEG_EVAL = False