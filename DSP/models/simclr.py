import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
from copy import deepcopy
from modeling.backbone.resnet import ResNet50




class SimCLR(nn.Module):
    def __init__(self, args):
        super(SimCLR, self).__init__()
        self.m_backbone = args.m_backbone
        # self.dense_head = args.dense_head
        self.m = args.m_update
        self.encoder_type = args.encoder
        self.dense_cl = args.dense_cl
        self.f = get_encoder(args.backbone, args.pre_train, args.output_stride, args.encoder)
        self.dense_neck = DenseCLNeck(in_channels=2048, hid_channels=512, out_channels=1, num_grid=None)

        self.pool = nn.AdaptiveAvgPool2d(1)


        # projection head
        self.g = nn.Sequential(
                                nn.Linear(2048, args.hidden_layer, bias=False),
                                nn.BatchNorm1d(args.hidden_layer),
                                nn.ReLU(inplace=True),
                                nn.Linear(args.hidden_layer, args.n_proj, bias=True)
                               )


        # Momentum Encoder
        if args.m_backbone:
            self.fm = deepcopy(self.f)
            self.gm = deepcopy(self.g)
            self.dense_m= deepcopy(self.dense_neck)
            for param in self.fm.parameters():
                param.requires_grad = False
            for param in self.gm.parameters():
                param.requires_grad = False
            for param in self.gm.parameters():
                param.requires_grad = False

    def forward(self, x, y=None):
        x, _ = self.f(x)
        if self.dense_cl:
            out_x = self.dense_neck(x)
        else:
            feat_x = self.pool(x)
            feat_x = torch.flatten(feat_x, start_dim=1)
            out_x = self.g(feat_x)
        if y is not None:
            if self.m_backbone:
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update()
                y, _ = self.fm(y)
                if self.dense_cl:
                    out_y = self.dense_neck(y)
                else:
                    feat_y = self.pool(y)
                    feat_y = torch.flatten(feat_y, start_dim=1)
                    out_y = self.gm(feat_y)
            else:
                y, _ = self.f(y)
                if self.dense_cl:
                    out_y = self.dense_neck(y)
                else:
                    feat_y = self.pool(y)
                    feat_y = torch.flatten(feat_y, start_dim=1)
                    out_y = self.g(feat_y)

            return x, y, out_x, out_y
        else:
            return F.normalize(feat_x, dim=-1), F.normalize(out_x, dim=-1)

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update of the key encoder
        """
        for param_f, param_fm in zip(self.f.parameters(), self.fm.parameters()):
            param_fm.data = param_fm.data * self.m + param_f.data * (1. - self.m)
        for param_g, param_gm in zip(self.f.parameters(), self.fm.parameters()):
            param_gm.data = param_gm.data * self.m + param_g.data * (1. - self.m)


class LinearEvaluation(nn.Module):
    """
    Linear Evaluation model
    """

    def __init__(self, n_features, n_classes):
        super(LinearEvaluation, self).__init__()
        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x1, x2):
        df = torch.abs(x1 - x2)
        return self.model(df)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_encoder(encoder, pre_train, output_stride, encoder_name):
    """
    Get Resnet backbone
    """

    class View(nn.Module):
        def __init__(self, shape=2048):
            super().__init__()
            self.shape = shape

        def forward(self, input):
            '''
            Reshapes the input according to the shape saved in the view data structure.
            '''
            batch_size = input.size(0)
            shape = (batch_size, self.shape)
            out = input.view(shape)
            return out

    def CMU_resnet50():

        if encoder_name=='resnet':
            resnet = ResNet50(BatchNorm=nn.BatchNorm2d, pretrained=pre_train, output_stride=output_stride)
            return resnet
        else:
             vgg16 = deeplab_V2()
             return vgg16


    return {
        'resnet18': torchvision.models.resnet18(pretrained=False),
        'resnet50': CMU_resnet50()
    }[encoder]

class DenseCLNeck(nn.Module):
    '''The non-linear neck in DenseCL.
        Single and dense in parallel: fc-relu-fc, conv-relu-conv
    '''

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None):
        super(DenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.BatchNorm2d(hid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))



    def forward(self, x):

        x = self.mlp2(x)  # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x)  # 1x1: bxdx1x1
        # x = x.view(x.size(0), x.size(1), -1)  # bxdxs^2
        # avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1)  # bxd
        return  x



if __name__ == "__main__":
    import torch
    model = SimCLR(a)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
