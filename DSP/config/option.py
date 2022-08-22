from util.utils import mkdir
from util.dist_util import init_distributed_mode
import argparse
import torch


class Options:
    def __init__(self):
        print("parsing..")
        parser = argparse.ArgumentParser(description="PyTorch Self-supervised Learning")
        parser.add_argument("--img_size", default=256, type=int, help="Image size(int) for RandomResizedCrop")  # 224, 96
        # SSL specific settings
        parser.add_argument("--ssl_epochs", default=300, type=int, help="Number of epochs for training SSL")  # 100, 200
        parser.add_argument("--ssl_model", default="simclr", type=str, help="SSL model")  # simclr simclr_cd
        parser.add_argument("--backbone", default="resnet50", type=str, help="SSL backbone")  # resnet18, resnet50
        parser.add_argument("--optimizer", default="lars", type=str, help="SSL optimizer")  # adam, lars
        parser.add_argument("--ssl_dataset", default="CMU", type=str, help="SSL training dataset")  # STL10, CIFAR100
        parser.add_argument("--ssl_batchsize", default=4, type=int, help="Batch size for SSL training ")  # 32, 64, 128
        parser.add_argument("--temperature", default=0.5, type=float, help="Temperature parameter for NTXent loss used for SSL training")
        parser.add_argument("--ssl_lr", default=0.0003, type=float, help="Learning rate for SSL training")  # 0.0003
        parser.add_argument("--n_proj", default=256, type=int, help="Projection head output size for SSL training")  # 64, 128
        parser.add_argument("--ssl_normalize", type=bool, default=True, help="Normalize projection head output for SSL training") # True, False
        parser.add_argument("--scheduler", type=bool, default=True, help="Use CosineAnnealingLR for SSL training")  # True, False
        parser.add_argument("--global_bn", type=bool, default=False, help="Use CosineAnnealingLR for SSL training")

        # zoom-in
        parser.add_argument("--zoom", type=bool, default=False, help="Whether to use zoom-in of cosine similarity in NT-Xent loss")
        parser.add_argument("--zoom_factor", default=10, type=int, help="Value of zoom-in in the zoom term")
        # Use margin in NT-Xent loss - EqCo
        parser.add_argument("--margin", type=bool, default=True, help="Whether to use margin in NT-Xent loss")
        parser.add_argument("--alpha", default=65536, type=int, help="Value of alpha in the margin term")

        # Use two backbones
        parser.add_argument("--m_backbone", type=bool, default=False, help="Whether to use momentum encoder")
        parser.add_argument("--m_update", type=float, default=0.990, help="Momentum update value (m)")  
        parser.add_argument("--output_stride", type=int, default=16, help="outputstride (8 or 16)")
        parser.add_argument("--pre_train", type=bool, default=False, help="pretrain_enc")  
        parser.add_argument("--encoder", type=str, default='resnet', help="resnet or vgg")  
        parser.add_argument("--dense_cl", type=bool, default=True, help="Whether to use dense prediction")  # True, False
        parser.add_argument("--copy_paste", type=bool, default=False, help="Whether to use copy paste aug")  # True, False
        parser.add_argument("--barlow_twins", type=bool, default=True, help="Whether to use copy paste aug")  # True, False
        parser.add_argument("--kd_loss", default=True, type=bool, help="kldiv")  # kl, rkd,sp,wasserstein,fitnet, rka, rkda, rkd-kl, rkda-kl
        parser.add_argument("--kd_loss_2", default="sp", type=str, help="diff kd losses:rkd,sp,fitnet, rkd,rka,rkda")  # kl, rkd,sp,wasserstein,fitnet, rka, rkda
        parser.add_argument("--alpha_kl", default=1000, type=float, help="Hyperparameter for KL-div")
        parser.add_argument("--alpha_sp", default=3000, type=float, help="Hyperparameter for similarity preserving")
        parser.add_argument("--alpha_inter_kd", default=100, type=float, help="Hyperparameter for inter and intra KL-div")
        parser.add_argument("--inter_kl", default=False, type=bool, help="calculate kl between to and t1 logits")  # kl, rkd,sp,wasserstein,fitnet, rka, rkda, rkd-kl, rkda-kl
        parser.add_argument(
            "--nodiff_tc", action="store_true", default=False, help="do not reset weight each generation"
        )

        parser.add_argument("--hidden_layer", type=int, default=512, help="hiddenlayer (512 or 1024)")
        parser.add_argument("--supervised_multihead", type=bool, default=True, help="Whether to use copy paste aug")  # True, False

        # Different weighted loss functions
        parser.add_argument("--criterion_weight", nargs="*", type=int, default=[1, 0, 0, 0],
                            help="Loss criterion weights for SSL training")  # [1, 1000, 0, 0], [1, 0, 25, 50]
        # Directory
        parser.add_argument("--data_dir", default="/data/input/datasets/VL-CMU-CD/pcd", type=str, help="Directory to import data")  # Absolute path
        parser.add_argument("--val_data_dir", default="/data/input/datasets/VL-CMU-CD/struc_test", type=str, help="Directory to import data")  # Absolute path

        parser.add_argument("--save_dir", default="/volumes1/tmp", type=str, help="Directory to save log and model")  # Absolute path /data/output/vijaya.ramkumar/sscd /volumes1/tmp  /sscdv2/runs_1
        # testing SSL model
        parser.add_argument("--test_dataset", default="CMU", type=str, help="Dataset for testing SSL methods")  # STL10, CIFAR10, ImageNet
        parser.add_argument("--test_data_dir", default="/data/input/datasets/VL-CMU-CD/struc_test", type=str, help="Directory to import data")  # Absolute path
        parser.add_argument("--linear_batchsize", default=16, type=int, help="Test batch size for linear evaluation")  # 32, 64, 128
        parser.add_argument("--linear_epochs", default=100, type=int, help="No.of epochs for Linear evaluation")  # 100, 200
        parser.add_argument("--linear_classes", default=1, type=int, help="No.of classes for Linear evaluation")  # 1 for binary classification
        parser.add_argument("--linear_lr", default=3e-4, type=float, help="Learning rate for Linear evaluation training")  # 0.0003

        # testing SSL model
        parser.add_argument("--sup_dataset", default="CIFAR100", type=str)  # STL10, CIFAR10, ImageNet
        parser.add_argument("--sup_data_dir", default="/volumes1/CIFAR100", type=str)  # Absolute path
        parser.add_argument("--sup_batchsize", default=256, type=int)  # 32, 64, 128
        parser.add_argument("--sup_lr", default=0.02, type=float)  # 0.0003
        parser.add_argument("--sup_epochs", default=100, type=int)  # 100

        # trained SSL model path
        parser.add_argument("--model_path", default=None, type=str, help="Saved SSL model path for transfer learning")  # Absolute path

        # Distributed
        parser.add_argument("--distribute", type=bool, default=False, help="Distributed Data Parallel")  # DistributedDataParallel
        parser.add_argument("--dist_url", type=str, default="env://")  # Default URL for DistributedDataParallel

        # Visualizing Heatmap for test Images
        parser.add_argument("--bestcheckpoint", default='/data/output/vijaya.ramkumar/sscd/runs/resnet50_bs_2/Wed_May_12_17:00:10_2021/checkpoint_model_170_model1.pth', type=str)  # '/data/output/vijaya.ramkumar/sscd/runs/resnet50_bs_8/Mon_Mar_22_16:59:24_2021/checkpoint_model_200_model1.pth'
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        mkdir(args.save_dir)
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.distribute:
            init_distributed_mode(args)
        if args.ssl_lr is None:
            # In SimCLR, linear LR scaling = 0.3 * args.batchsize / 256 and square root LR scaling  0.075 × math.sqrt(BatchSize)
            args.ssl_lr = 0.03 * args.ssl_batchsize / 256
            # args.ssl_lr = 0.075 * math.sqrt(args.ssl_batchsize)
        return args
