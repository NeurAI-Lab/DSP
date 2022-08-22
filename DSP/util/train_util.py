import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os
from transforms.simclr_transform import SimCLRTransform
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from util.utils import save_checkpoint, log
from criterion.ntxent import NTXent, BarlowTwinsLoss_CD
from criterion.sim_preserving_kd import criterion_MSE,distillation,fitnet_loss,similarity_preserving_loss, RKD, similarity_preserving_loss_cd, JSD
import dataset.CMU as CMU
import dataset.PCD as PCD

def get_criteria(args):
    """
    Loss criterion / criteria selection for training
    """
    if args.barlow_twins :
        criteria = {'Barlow': [BarlowTwinsLoss_CD(args)]}   #BarlowTwinsLoss
    else:
        criteria = {'ntxent': [NTXent(args), args.criterion_weight[0]]}

    return criteria



def write_scalar(writer, total_loss,total_loss_bl, total_loss_kd,total_loss_sp, loss_p_c, leng, epoch):
    """
    Add Loss scalars to tensorboard
    """
    writer.add_scalar("Total_Loss/train", total_loss/leng,epoch)
    writer.add_scalar("Total_Loss_bl",  total_loss_bl/leng,epoch)
    writer.add_scalar("Total_kd loss/train", total_loss_kd/leng, epoch)
    writer.add_scalar("Total_sp loss/train", total_loss_sp/leng, epoch)


    for k in loss_p_c:
        writer.add_scalar("{}_Loss/train".format(k), loss_p_c[k] / leng, epoch)


def trainloaderSimCLR(args):
    """
    Load training data through DataLoader
    """
    transform = SimCLRTransform(args.img_size)

    if args.ssl_dataset == 'CMU':
        DATA_PATH = os.path.join(args.data_dir)

        VAL_DATA_PATH = os.path.join(args.val_data_dir)


        train_dataset = CMU.Dataset(DATA_PATH,
                                         'train', 'ssl', transform= False,     #ssl
                                         transform_med = transform)
        # test_dataset = CMU.Dataset(VAL_DATA_PATH, 'val', transform=False,
        #                         transform_med=None)
    elif args.ssl_dataset == 'PCD':
        print('PCD dataset loaded')
        DATA_PATH = os.path.join(args.data_dir)

        VAL_DATA_PATH = os.path.join(args.val_data_dir)

        train_dataset = PCD.Dataset(DATA_PATH,
                                    'train', 'ssl', transform=False,  # ssl
                                    transform_med=transform)
    #  Data Loader
    if args.distribute:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.ssl_batchsize,sampler=train_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.ssl_batchsize, shuffle=True, drop_last=True)
        # val_loader = DataLoader(train_dataset, batch_size=args.ssl_batchsize, shuffle=True, drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

    log("Took {} time to load data!".format(datetime.now() - args.start_time))
    return train_loader

def various_distance( out_vec_t0, out_vec_t1, dist_flag='l2'):

    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=1)
    if dist_flag == 'cos':
        similarity = F.cosine_similarity(out_vec_t0, out_vec_t1)
        distance = 1 - 2 * similarity / np.pi
    return distance

def train_one_epoch(args, train_loader, model, criteria, optimizer, scheduler, epoch):
    """
    Train one epoch of SSL model

    """
    # torch.autograd.set_detect_anomaly(True)
    loss_per_criterion = {}
    total_loss = 0
    total_sup_loss = 0
    total_loss_bl = 0
    total_loss_kd = 0
    total_loss_sp = 0

    for i, batch in enumerate(train_loader):
        p1, p2, n1, n2, f1,f2, label = batch    # x, y = positive pair belonging to t0 images ; x1,y1 = positive pair belonging to t1 images
        p1 = p1.cuda(device=args.device)
        p2 = p2.cuda(device=args.device)
        n1 = n1.cuda(device=args.device)
        n2 = n2.cuda(device=args.device)
        label = label.cuda(device=args.device)
        label = label.float()
        optimizer.zero_grad()
        if args.barlow_twins == True:
            if args.dense_cl==True:
                xe, ye, zx, zy = model(p1, p2)
                x1e, y1e, zx1, zy1 = model(n1, n2)
                diff_feat0= torch.nn.functional.pairwise_distance(zx, zx1)
                diff_feat1 = torch.nn.functional.pairwise_distance(zy , zy1)
                diff_feat2 = torch.nn.functional.pairwise_distance(zx , zy1)
                diff_feat3 = torch.nn.functional.pairwise_distance(zy , zx1)
            else:
                xe, ye, zx, zy = model(p1, p2)
                x1e, y1e, zx1, zy1 = model(n1, n2)
                ## simple diff layer to get change map
                diff_feat0 = torch.abs(zx - zx1)
                diff_feat1 = torch.abs(zy - zy1)
                diff_feat2 = torch.abs(zx - zy1)
                diff_feat3 = torch.abs(zy - zx1)
        else:
            _, _, zx, zy = model(p1, p2)
            _, _, zx1, zy1 = model(n1, n2)
        # Multiple loss aggregation
        loss = torch.tensor(0).to(args.device)
        for k in criteria:
            global_step = epoch * len(train_loader) + i
            if args.barlow_twins == True:
                if args.nodiff_tc:
                    loss_bl = criteria[k][0](zx, zx1, diff_feat2, diff_feat3)
                else:
                    loss_bl = criteria[k][0](diff_feat0, diff_feat1, diff_feat2, diff_feat3 )

                if args.kd_loss==True:
                    jsd = JSD(args)
                    # loss_kd_1 = distillation(zx, zy, T=4)
                    loss_kd_1 = jsd(zx, zy)
                    loss_kd_2 = jsd(zx1, zy1)
                    intra_kd_loss = loss_kd_1 + loss_kd_2
                    loss_sp = 0
                    if args.inter_kl==True:
                        loss_kd_3 = jsd(zx, zx1)
                        loss_kd_4 = jsd(zy, zy1)
                        inter_kd_loss = loss_kd_3 + loss_kd_4
                        loss_kd = (args.alpha_kl * intra_kd_loss) + (args.alpha_inter_kd*inter_kd_loss)
                    else:
                        loss_kd = args.alpha_kl * intra_kd_loss
                    loss = loss_bl + loss_kd
                    if args.kd_loss_2 == 'fitnet':
                        loss_ft_1 = fitnet_loss(A_t=xe, A_s=ye, rand=False, noise=0.1)
                        loss_ft_2 = fitnet_loss(A_t=x1e, A_s=y1e, rand=False, noise=0.1)
                        loss_sp = (args.alpha_sp*(loss_kd_1 + loss_kd_2))
                        loss = loss_bl + loss_kd + loss_sp
                    elif args.kd_loss_2 == 'sp':
                        loss_sp = ((args.alpha_sp)* similarity_preserving_loss_cd(xe, x1e, ye, y1e))
                        loss = loss_bl + loss_kd + loss_sp
            else:

                loss = loss_bl

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % 50 == 0:

            log("Batch {}/{}. Loss: {}. Loss_bl: {}. Loss_kd: {}. Loss_sp{}. Time elapsed: {} ".format(i, len(train_loader), loss.item(),loss_bl.item(),loss_kd.item()
                                                                                      ,loss_sp.item(),datetime.now() - args.start_time))
        total_loss += loss.item()
        total_loss_bl += loss_bl.item()
        total_loss_kd += loss_kd.item()
        total_loss_sp += loss_sp.item()

    return total_loss, total_loss_bl, total_loss_kd, total_loss_sp, loss_per_criterion





def trainSSL(args, model, optimizer, criteria, writer, scheduler=None):
    """
    Train a SSL model
    """
    if not args.visualize_heatmap :
        model.train()
        # Data parallel Functionality
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            log('Model converted to DataParallel model with {} cuda devices'.format(torch.cuda.device_count()))
        model = model.to(args.device)

        train_loader = trainloaderSimCLR(args)

        for epoch in tqdm(range(1, args.ssl_epochs + 1)):
            total_loss, total_loss_bl, total_loss_kd,total_loss_sp, loss_per_criterion = train_one_epoch(args, train_loader, model, criteria, optimizer, scheduler, epoch)

            write_scalar(writer, total_loss,total_loss_bl, total_loss_kd,total_loss_sp, loss_per_criterion, len(train_loader), epoch)
            log("Epoch {}/{}. Total Loss: {}.   Time elapsed: {} ".
                format(epoch, args.ssl_epochs, total_loss / len(train_loader), total_loss_bl / len(train_loader),total_loss_kd / len(train_loader), datetime.now() - args.start_time))


            # Save checkpoint after every epoch
            path = save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch, filename='checkpoint.pth'.format(epoch))
            if os.path.exists:
                state_dict = torch.load(path, map_location=args.device)
                model.load_state_dict(state_dict)

            # Save the model at specific checkpoints
            if epoch % 10 == 0:

                if torch.cuda.device_count() > 1:
                    save_checkpoint(state_dict=model.module.state_dict(), args=args, epoch=epoch,
                                    filename='checkpoint_model_{}_model1.pth'.format(epoch))

                else:
                    save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch,
                                    filename='checkpoint_model_{}_model1.pth'.format(epoch))

        log("Total training time {}".format(datetime.now() - args.start_time))


    writer.close()
