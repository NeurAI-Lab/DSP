import torch
import torch.nn as nn
from util.utils import positive_mask
import os
import math
import util.utils as utils
import torch.nn.functional as F


class NTXent(nn.Module):
    """
    The Normalized Temperature-scaled Cross Entropy Loss
    Source: https://github.com/Spijkervet/SimCLR
    """

    def __init__(self, args):
        super(NTXent, self).__init__()
        self.batch_size = args.ssl_batchsize
        self.margin = args.margin
        self.alpha = args.alpha
        self.temperature = args.temperature
        self.device = args.device
        self.mask = positive_mask(args.ssl_batchsize)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.N = 4 * self.batch_size
        self.zoom = args.zoom
        self.zoom_factor = args.zoom_factor
        self.writer = args.writer


    def forward(self, zx, zy, zx1, zy1,  global_step):
        """
        zx: projection output of batch zx
        zy: projection output of batch zy
        :return: normalized loss
        """
        positive_samples, negative_samples = self.sample_no_dict(zx, zy, zx1, zy1)
        if self.margin:
            m = self.temperature * math.log(self.alpha / negative_samples.shape[1])
            positive_samples = ((positive_samples * self.temperature) - m) / self.temperature

        labels = torch.zeros(self.N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= self.N

        return loss


    def sample_no_dict(self, zx, zy,  zx1, zy1):
        """
        Negative samples without dictionary
        """
        # print(zx.shape)
        z = torch.cat((zx, zy, zx1,zy1), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # print(sim.shape,self.batch_size )

        # Splitting the matrix into 4 blocks so as to count number of positive and negative samples
        sim_left, sim_right = torch.chunk(sim, 2, dim=1)
        sim_lu,sim_ll = torch.chunk(sim_left, 2, dim=0)
        sim_ru,sim_rl = torch.chunk(sim_right, 2, dim=0)
        # print(sim_lu.shape,self.batch_size )

        # Extract positive samples from each block
        #sim_xy = torch.diag(sim, self.batch_size)
        pos_1 = torch.diag(sim_lu, self.batch_size)
        pos_2 = torch.diag(sim_lu, -self.batch_size)
        pos_3 = torch.diag(sim_rl, self.batch_size)
        pos_4 = torch.diag(sim_rl, -self.batch_size)
        # sim_yx = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((pos_1, pos_2, pos_3, pos_4), dim=0).reshape(self.N, 1)

        # Extract negative samples
        neg_lu = sim_lu[self.mask].reshape(self.batch_size*2, 2*(self.batch_size-1) )
        neg_rl = sim_rl[self.mask].reshape(self.batch_size*2, 2*(self.batch_size-1))

        # Concatenating the extracted negatives from sim block left upper and right lower.
        neg_u = torch.cat((neg_lu, sim_ru), dim=1)
        neg_l = torch.cat((sim_ll, neg_rl), dim=1)
        negative_samples = torch.cat((neg_u, neg_l), dim=0)

        return positive_samples, negative_samples



class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, device, lambda_param=5e-3):                                                                                          
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):                                                                                            
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD                                                                                                                                                                                                                                                           
        z_a_norm = z_a_norm.view(z_a_norm.size(0), z_a_norm.size(1)* z_a_norm.size(2))
        z_b_norm = z_b_norm.view(z_b_norm.size(0), z_b_norm.size(1)*z_b_norm.size(2))

        N = z_a.size(0)
        # D = z_a.size(1)
        D = z_a_norm.size(1)

        # print (z_a_norm.T.shape, z_b_norm.shape)
        # cross-correlation matrix
        # c= torch.einsum('yxb,bxy->xy', (z_a_norm.T, z_b_norm))
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # print (c.shape)

        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss


class BarlowTwinsLoss_CD(torch.nn.Module):
    def __init__(self, args, lambda_param=5e-3):
        super(BarlowTwinsLoss_CD, self).__init__()
        self.lambda_param = lambda_param
        self.device = args.device
        self.dense_cl = args.dense_cl

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor,z_c: torch.Tensor, z_d: torch.Tensor):   #, z_c: torch.Tensor, z_d: torch.Tensor

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD
        z_c_norm = (z_c - z_c.mean(0)) / z_c.std(0)  # NxD
        z_d_norm = (z_d - z_d.mean(0)) / z_d.std(0)  # NxD

        N = z_a.size(0)
        if self.dense_cl == True:
        ## for dense activation
            z_a_norm = z_a_norm.view(z_a_norm.size(0),  -1)
            z_b_norm = z_b_norm.view(z_b_norm.size(0),  -1)
            z_c_norm = z_c_norm.view(z_c_norm.size(0), -1)
            z_d_norm = z_d_norm.view(z_d_norm.size(0), -1)
            D = z_a_norm.size(1)

        else:
            D = z_a.size(1)
        # print(z_a_norm.shape)
        # cross-correlation matrix
        c1 = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # c2 = torch.mm(z_c_norm.T, z_d_norm) / N # DxD


        # loss
        c_diff1 = (c1 - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff1[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss1 = c_diff1.sum()

        loss = loss1
        return loss

