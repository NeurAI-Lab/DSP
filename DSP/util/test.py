from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, STL10
import os
import numpy as np
from config.option import Options
from util.utils import summary_writer, logger, eval_image_rewrite,pxEval_maximizeFMeasure, init_metric_for_class_for_cmu
from models.simclr import LinearEvaluation
from util.utils import log, save_checkpoint
from transforms.simclr_transform import SimCLRTransform
from optimizers.lars import LARC
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import dataset.CMU as CMU
from torch.nn import functional as F
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def initialize():
    """
    Initialize the test script by parsing arguments, create summary writer and logger
    """
    args = Options().parse()
    log_dir = os.path.dirname(os.path.abspath(args.model_path))
    _, checkpoint = os.path.split(args.model_path)
    writer = summary_writer(args, log_dir, checkpoint + '_Evaluation')
    logger(args, checkpoint + '{}_test.log'.format(args.test_dataset))
    args.start_time = datetime.now()
    log("Starting testing of SSL model at  {}".format(datetime.now()))
    log("arguments parsed: {}".format(args))
    return args, writer


def get_domain_net(args):
    """ DomainNet datasets - QuickDraw, Sketch, ClipArt """

    DATA_PATH = os.path.join(args.data_dir)
    # TRAIN_LABEL_PATH = os.path.join(args.data_dir)
    # TRAIN_TXT_PATH = os.path.join(TRAIN_DATA_PATH, 'train_pair.txt')
    #
    # VAL_DATA_PATH = os.path.join(args.val_data_dir)
    # VAL_LABEL_PATH = os.path.join(args.val_data_dir)
    # VAL_TXT_PATH = os.path.join(VAL_DATA_PATH, 'test_pair.txt')

    train_dataset = CMU.Dataset(DATA_PATH,'train', 'linear_eval', transform=False,     #ssl
                                     transform_med =SimCLRTransform(args.img_size))
    if args.visualize_heatmap:

        train_dataset = CMU.Dataset(DATA_PATH, 'train',  transform=True,  # ssl
                                    transform_med=None)
        test_dataset = CMU.Dataset(DATA_PATH, 'val', transform=True, # original dataset
                                     transform_med=None)   #.test_transform SimCLRTransform(args.img_size).test_transform
    else:
        test_dataset = CMU.Dataset(DATA_PATH, 'val', 'linear_eval', transform=False,
                               transform_med=SimCLRTransform(args.img_size))
    return train_dataset, test_dataset


def testloaderSimCLR(args, dataset, transform, batchsize, data_dir, val_split=0.15):
    """
    Load test datasets
    """
    if dataset == 'CIFAR100':
        train_d = CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_d = CIFAR100(data_dir, train=False, download=True, transform=transform)
    elif dataset == 'CIFAR10':
        train_d = CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_d = CIFAR10(data_dir, train=False, download=True, transform=transform)
    elif dataset == 'CMU' or dataset == 'CDnet' or dataset == 'PCD':
        train_d, test_d = get_domain_net(args)

    # train - validation split
    val_size = int(val_split * len(train_d))
    train_size = len(train_d) - val_size
    train_d, val_d = random_split(train_d, [train_size, val_size])

    if args.distribute:
        train_sampler = DistributedSampler(train_d)
        val_sampler = DistributedSampler(val_d)
        test_sampler = DistributedSampler(test_d)
        train_loader = DataLoader(train_d, batch_size=batchsize,sampler=train_sampler, drop_last=True)
        val_loader = DataLoader(val_d, batch_size=batchsize, sampler=val_sampler, drop_last=True)
        test_loader = DataLoader(test_d, batch_size=batchsize, sampler=test_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_d, batch_size=batchsize, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_d, batch_size=batchsize, shuffle=True, drop_last=True)
        if args.visualize_heatmap:
            train_loader = DataLoader(train_d, batch_size=1, shuffle=False, drop_last=True)

            test_loader = DataLoader(test_d, batch_size=1, shuffle=False, drop_last=True)
        else:
            test_loader = DataLoader(test_d, batch_size=batchsize, shuffle=False, drop_last=True)

    log("Took {} time to load data!".format(datetime.now() - args.start_time))
    return train_loader, val_loader, test_loader


def binary_acc(y_pred, y):
    ''' For calculating  accuracy in a binary classification problem'''
    y_pred_round = torch.round(torch.sigmoid(y_pred))
    correct_res_sum = (y_pred_round == y).sum().float()
    accuracy = correct_res_sum/y.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy
def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def single_layer_similar_heatmap_visual(output_t0,output_t1, dist_flag):

    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    similar_distance_map_rz = nn.functional.interpolate(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]),size=[256,256], mode='bilinear',align_corners=True)
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    cv2.imshow('image', similar_dis_map_colorize)
    cv2.waitKey(0)
    return similar_distance_map_rz.data.cpu().numpy(), similar_dis_map_colorize

def train_or_val(args, loader, simclr, model, criterion, optimizer=None, scheduler=None, train=False):
    """
    Train Linear model
    """
    loss_epoch = 0
    accuracy_epoch = 0
    simclr.eval()
    if train:
        model.train()
    else:
        model.eval()
        model.zero_grad()

    for step, (x1, x2, y) in enumerate(loader):
        x1 = x1.to(args.device)
        x2 = x2.to(args.device)
        y = y.to(args.device)

        x1 = simclr.f(x1)
        x2 = simclr.f(x2)
        f1 = torch.flatten(x1, start_dim=1)
        f2 = torch.flatten(x2, start_dim=1)
        # single_layer_similar_heatmap_visual(x1,x2, 'l2')
        logits = model(f1, f2)
        y = y.unsqueeze(1)
        y = y.float()
        loss = criterion(logits, y)
        acc = binary_acc(logits, y)
        # predicted = output.argmax(1)
        # acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        loss_epoch += loss.item()
    return loss_epoch, accuracy_epoch


def testSSL(args, writer, simclr):
    for param in simclr.parameters():
        param.requires_grad = False
    linear_model = LinearEvaluation(2048, args.linear_classes)
    linear_model = linear_model.cuda()

    optimizer = optim.Adam(linear_model.parameters(), lr=0.0003, weight_decay=1e-6)
    scheduler = None
    if args.optimizer == 'lars':
        optimizer_ = optim.SGD(linear_model.parameters(), lr=args.ssl_lr)
        optimizer = LARC(optimizer_)
        if args.scheduler:
            scheduler = CosineAnnealingLR(optimizer_, T_max=100, eta_min=3e-4)

    if args.distribute:
        if args.global_bn:
            simclr = nn.SyncBatchNorm.convert_sync_batchnorm(simclr)
            linear_model = nn.SyncBatchNorm.convert_sync_batchnorm(linear_model)
        simclr = DDP(simclr, device_ids=[args.gpu])
        linear_model = DDP(linear_model, device_ids=[args.gpu])
    transform = SimCLRTransform(args.img_size).test_transform
    train_loader, val_loader, test_loader = testloaderSimCLR(args, args.test_dataset, transform, args.linear_batchsize, args.test_data_dir)
    loss_criterion = nn.BCEWithLogitsLoss()                # nn.CrossEntropyLoss()

    best_acc = 0.0
    log('Testing SSL Model on {}.................'.format(args.test_dataset))
    _, ck_name = os.path.split(args.model_path)
    for epoch in range(1, args.linear_epochs + 1):
        loss_epoch, accuracy_epoch = train_or_val(args, train_loader, simclr, linear_model, loss_criterion, optimizer, scheduler, train=True)
        log(f"Epoch [{epoch}/{args.linear_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}")

        loss_epoch1, accuracy_epoch1 = train_or_val(args, val_loader, simclr, linear_model, loss_criterion, train=False)
        val_accuracy = accuracy_epoch1 / len(test_loader)
        log(f"Epoch [{epoch}/{args.linear_epochs}] \t Validation accuracy {val_accuracy}")

        if best_acc < val_accuracy:
            best_acc = val_accuracy
            log('Best accuracy achieved so far: {}'.format(best_acc))
            if args.distribute:
                # Save DDP model's module
                save_checkpoint(state_dict=linear_model.module.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_best_linear_model_{}_{}.pth'.format(args.test_dataset, ck_name))
            else:
                save_checkpoint(state_dict=linear_model.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_best_linear_model_{}_{}.pth'.format(args.test_dataset, ck_name))
        writer.add_scalar("Accuracy/train{}".format(args.test_dataset), accuracy_epoch / len(train_loader), epoch)
        writer.add_scalar("Accuracy/val{}".format(args.test_dataset), accuracy_epoch1 / len(test_loader), epoch)

    # Load best linear model and run inference on test set
    state_dict = torch.load(os.path.join(args.log_dir, 'checkpoint_best_linear_model_{}_{}.pth'.format(args.test_dataset, ck_name)), map_location=args.device)
    linear_best_model = LinearEvaluation(2048, args.linear_classes)
    linear_best_model.load_state_dict(state_dict)
    linear_best_model = linear_best_model.cuda()
    test_loss, test_acc = train_or_val(args, test_loader, simclr, linear_best_model, loss_criterion, train=False)
    test_acc = test_acc / len(test_loader)
    log(f" Test accuracy : {test_acc}")
    writer.add_text("Test Accuracy {} :".format(args.test_dataset), "{}".format(test_acc))


def test_all_datasets(args, writer, model):
    """
    Test all datasets for linear evaluation
    """
    # CIFAR10
    args.img_size = 256
    args.test_dataset = "CMU"
    args.test_data_dir = "/data/input/datasets/VL-CMU-CD/struc_test"
    args.linear_classes = 1
    testSSL(args, writer, model)


def visualize(args, model):
    """
    Train Linear model
    """
    if args.bestcheckpoint:
        checkpt = torch.load(args.bestcheckpoint)
        model.load_state_dict(checkpt)
        print(model.f.conv1.weight)
    else:
        raise RuntimeError('provide best checkpoint')
    train_loader, val_loader, test_loader = testloaderSimCLR(args, args.test_dataset, transform=None,batchsize= 1, data_dir=args.val_data_dir)
    embed1= []
    embed2= []

    for step, (x1, x2, y, _) in enumerate(train_loader):
        x1, _ = model.f(x1)
        x2, _ = model.f(x2)
        x1 = F.avg_pool2d(x1, kernel_size = (96,128), stride = 1)
        x2 = F.avg_pool2d(x2, kernel_size = (96,128), stride = 1)
        f1 = torch.flatten(x1, start_dim=1)
        f2 = torch.flatten(x2, start_dim=1)
        embed1.append(f1.detach().numpy())
        embed2.append(f2.detach().numpy())
    embed1.extend(embed2)
    embed = np.asarray(embed1)
    embed=  np.squeeze(embed, 1)
    tsne = TSNE(n_components=2, random_state=1, n_iter=1000, metric='cosine', perplexity=10)
    features = tsne.fit_transform(embed)
    pos_x = features[0:len(train_loader),0]
    pos_y = features[0:len(train_loader),1]
    neg_x = features[len(train_loader):, 0]
    neg_y = features[len(train_loader):, 1]
    FS = (10, 8)
    n = range(0,len(train_loader))
    fig, ax = plt.subplots(figsize=FS)
    ax.scatter(pos_x, pos_y, s=10, c='b', marker="o", label='pos')
    ax.scatter(neg_x, neg_y, s=10, c='r', marker="x", label='neg')
    for i, txt in enumerate(n):
        ax.annotate(txt, (pos_x[i], pos_y[i]))
        ax.annotate(txt, (neg_x[i], neg_y[i]))
    plt.show()
    plt.pause(1)

def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def single_layer_similar_heatmap_visual(output_t0,output_t1, dist_flag = 'l2'):

    # interp = nn.functional.interpolate(size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], mode='bilinear')
    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()

    similar_distance_map_rz = nn.functional.interpolate(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]),size=[512,512], mode='bilinear',align_corners=True)
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    # cv2.imshow('ImageWindow', similar_dis_map_colorize)
    # cv2.waitKey()
    # cv2.imwrite(save_weight_fig_dir, similar_dis_map_colorize)
    return similar_distance_map_rz.data.cpu().numpy()

def visualize_heatmap(args, model):
    """
    Train Linear model
    """


    if args.bestcheckpoint:
        checkpt = torch.load(args.bestcheckpoint)
        model.load_state_dict(checkpt)
    else:
        raise RuntimeError('provide best checkpoint')
    train_loader, val_loader, test_loader = testloaderSimCLR(args, args.test_dataset, transform=True, batchsize=1,
                                                             data_dir=args.val_data_dir)

    model.eval()
    with torch.no_grad():
        cont_conv5_total, cont_fc_total, cont_embedding_total, num = 0.0, 0.0, 0.0, 0.0
        metric_for_conditions = init_metric_for_class_for_cmu(1)
        count = 0
    for step, (x1, x2, y) in enumerate(test_loader):
        ## input size is hardcoded
        input1, input2, targets = x1.to(args.device), x2.to(args.device), y.to(args.device)
        count = count +1
        print(count)
        # input1, input2, targets = input1.to(args.device), input2.to(args.device), targets.to(args.device)

        # out_conv5_p1, out_fc_p1, out_embedding_p1 = model(input1, input2)
        # out_embedding_t0, out_embedding_t1 = out_embedding_p1

        out_dense_p1, _ = model.f(x1)
        out_dense_p2, _ = model.f(x2)
        out_dense_p1, out_fc_p1, out_conv_p1 = model.g(out_dense_p1)
        # out_dense_p2 = model.g(out_dense_p2)
        out_dense_p2, out_fc_p2, out_conv_p2 = model.g(out_dense_p2)

        out_t0_embedding_norm, out_t1_embedding_norm = model.norm(out_dense_p1, 2, dim=1), model.norm(out_dense_p2, 2, dim=1)
        out_t0_fc_norm, out_t1_fc_norm = model.norm(out_fc_p1, 2, dim=1), model.norm(out_fc_p2, 2, dim=1)
        out_t0_conv_norm, out_t1_conv_norm = model.norm(out_conv_p1, 2, dim=1), model.norm(out_conv_p2, 2, dim=1)

        embedding_distance_map = single_layer_similar_heatmap_visual(out_t0_embedding_norm, out_t1_embedding_norm)
        fc_distance_map = single_layer_similar_heatmap_visual(out_t0_fc_norm, out_t1_fc_norm)
        conv_distance_map = single_layer_similar_heatmap_visual(out_t0_conv_norm, out_t1_conv_norm)

        prob_change = embedding_distance_map[0][0]
        gt = targets.data.cpu().numpy()
        FN, FP, posNum, negNum = eval_image_rewrite(gt[0][0], prob_change, cl_index=1)

        metric_for_conditions[0]['total_fp'] += FP
        metric_for_conditions[0]['total_fn'] += FN
        metric_for_conditions[0]['total_posnum'] += posNum
        metric_for_conditions[0]['total_negnum'] += negNum


    thresh = np.array(range(0, 256)) / 255.0
    conds = metric_for_conditions.keys()
    for cond_name in conds:
        # print(cond_name)
        total_posnum = metric_for_conditions[cond_name]['total_posnum']
        total_negnum = metric_for_conditions[cond_name]['total_negnum']
        total_fn = metric_for_conditions[cond_name]['total_fn']
        total_fp = metric_for_conditions[cond_name]['total_fp']
        metric_dict = pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                                 total_fn, total_fp, thresh=thresh)
        print(total_posnum, total_negnum)
        metric_for_conditions[cond_name].setdefault('metric', metric_dict)

    f_score_total = 0.0
    for cond_name in conds:
        pr, recall, f_score = metric_for_conditions[cond_name]['metric']['precision'], \
                              metric_for_conditions[cond_name]['metric']['recall'], \
                              metric_for_conditions[cond_name]['metric']['MaxF']

        f_score_total += f_score

    print('score', f_score_total / (len(conds)))
    return f_score_total / len(conds)






