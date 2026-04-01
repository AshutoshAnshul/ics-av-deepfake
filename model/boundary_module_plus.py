# originally from https://github.com/ControlNet/LAV-DF/blob/master/model/boundary_module_plus.py


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Sequential, LeakyReLU

from model.boundary_module import BoundaryModule
from utils import Conv2d, Conv1d


class ConvUnit(nn.Module):
    """
    Unit in NestedUNet
    """

    def __init__(self, in_ch, out_ch, is_output=False):
        super(ConvUnit, self).__init__()
        module_list = [nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)]
        if is_output is False:
            module_list.append(nn.BatchNorm1d(out_ch))
            module_list.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*module_list)

    def forward(self, x):
        x = self.conv(x)
        return x


class NestedUNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=400, out_ch=2):
        super(NestedUNet, self).__init__()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2)

        n1 = 512
        filters = [n1, n1 * 2, n1 * 3]
        self.conv0_0 = ConvUnit(in_ch, filters[0], is_output=False)
        self.conv1_0 = ConvUnit(filters[0], filters[0], is_output=False)
        self.conv2_0 = ConvUnit(filters[0], filters[0], is_output=False)

        self.conv0_1 = ConvUnit(filters[1], filters[0], is_output=False)
        self.conv1_1 = ConvUnit(filters[1], filters[0], is_output=False)

        self.conv0_2 = ConvUnit(filters[2], filters[0], is_output=False)

        self.final = nn.Conv1d(filters[0] * 3, out_ch, kernel_size=1)
        # self.final = ConvUnit(filters[0] * 3, out_ch, is_output=True)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        out_feature = torch.cat([x0_0, x0_1, x0_2], 1)  # for calculating loss
        final_feature = self.final(out_feature)
        out = self.out(final_feature)

        return out, out_feature


class PositionAwareAttentionModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=None, dim=2):
        super(PositionAwareAttentionModule, self).__init__()

        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.dim = dim

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if self.dim == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.InstanceNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2,))
            bn = nn.InstanceNorm2d

        self.g = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.phi = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        if self.sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)
        # value
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # query
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # key
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        y = torch.matmul(theta_x, phi_x)
        y = F.softmax(y, dim=2)

        y = torch.matmul(y, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)

        y = y + x
        return y


class ChannelAwareAttentionModule(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dim=2):
        super(ChannelAwareAttentionModule, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.dim = dim

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if self.dim == 2:
            conv_nd = nn.Conv2d
            bn = nn.InstanceNorm2d
        else:
            conv_nd = nn.Conv1d
            bn = nn.InstanceNorm2d

        self.g = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.phi = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0),
            bn(self.inter_channels),
            nn.ReLU(inplace=True)
        )
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        if torch.isnan(g_x).any():
            print('nan in g_x')

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        if torch.isnan(theta_x).any():
            print('nan in theta_x')

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.permute(0, 2, 1)
        if torch.isnan(phi_x).any():
            print('nan in phi_x')

        f = torch.matmul(theta_x, phi_x)
        if torch.isnan(f).any():
            print('nan in y_matmul_1')

        y = F.softmax(f, dim=2)
        if torch.isnan(y).any():
            print('nan in y_softmax')
            print(f.max(), f.min())

        y = torch.matmul(y, g_x)
        if torch.isnan(y).any():
            print('nan in y_matmul_2')

        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)
        if torch.isnan(y).any():
            print('nan in y_W')

        y = y + x
        return y


def conv_block(in_ch, out_ch, kernel_size=3, stride=1, bn_layer=False, activate=False):
    module_list = [nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=1)]
    if bn_layer:
        module_list.append(nn.BatchNorm2d(out_ch))
        module_list.append(nn.ReLU(inplace=True))
    if activate:
        module_list.append(nn.Sigmoid())
    conv = nn.Sequential(*module_list)
    return conv


class ProposalRelationBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=128, out_channels=2, sub_sample=False):
        super(ProposalRelationBlock, self).__init__()
        self.p_net = PositionAwareAttentionModule(in_channels, inter_channels=inter_channels, sub_sample=sub_sample, dim=2)
        self.c_net = ChannelAwareAttentionModule(in_channels, inter_channels=inter_channels, dim=2)
        self.conv0_0 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)
        self.conv0_1 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)

        self.conv1 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)
        self.conv2 = conv_block(in_channels, out_channels, 3, 1, bn_layer=False, activate=True)
        self.conv3 = conv_block(in_channels, out_channels, 3, 1, bn_layer=False, activate=True)
        self.conv4 = conv_block(in_channels, in_channels, 3, 1, bn_layer=True, activate=False)
        self.conv5 = conv_block(in_channels, out_channels, 3, 1, bn_layer=False, activate=True)

    def forward(self, x):
        x_p = self.conv0_0(x)
        x_c = self.conv0_1(x)

        x_p = self.p_net(x_p)
        x_c = self.c_net(x_c)

        x_p_0 = self.conv1(x_p)
        x_p_1 = self.conv2(x_p_0)

        x_c_0 = self.conv4(x_c)
        x_c_1 = self.conv5(x_c_0)

        x_p_c = self.conv3(x_p_0 + x_c_0)
        return x_p_1, x_c_1, x_p_c


class BoundaryModulePlus(BoundaryModule):
    def __init__(self, n_feature_in, n_features=(512, 128), num_samples: int = 10, temporal_dim: int = 512,
        max_duration: int = 40
    ):
        super().__init__(n_feature_in, n_features, num_samples, temporal_dim, max_duration)
        del self.block1
        dim0, dim1 = n_features
        # (B, dim0, max_duration, temporal_dim) -> (B, max_duration, temporal_dim)
        self.block1 = Sequential(
            Conv2d(dim0, dim1, kernel_size=1, build_activation=LeakyReLU),
            Conv2d(dim1, dim1, kernel_size=3, padding=1, build_activation=LeakyReLU)
        )
        # Proposal Relation Block in BSN++ mechanism
        self.proposal_block = ProposalRelationBlock(dim1, dim1, 1, sub_sample=True)
        self.out = Rearrange("b c d t -> b (c d) t")

    def forward(self, feature: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        confidence_map = self.bm_layer(feature)
        confidence_map = self.block0(confidence_map)
        confidence_map = self.block1(confidence_map)
        confidence_map_p, confidence_map_c, confidence_map_p_c = self.proposal_block(confidence_map)

        confidence_map_p = self.out(confidence_map_p)
        confidence_map_c = self.out(confidence_map_c)
        confidence_map_p_c = self.out(confidence_map_p_c)
        return confidence_map_p, confidence_map_c, confidence_map_p_c

# -------------------------------------------- Boundary Module Fusion --------------------------------------

class ModalFeatureAttnBoundaryMapFusion(nn.Module):
    """
    Fusion module for video and audio boundary maps.

    Input:
        F_v: (B, C_f, T)
        F_a: (B, C_f, T)
        F_f: (B, C_f, T)
        M_v^: (B, D, T)
        M_a^: (B, D, T)
        M_f^: (B, D, T)

    Output:
        M^: (B, D, T)
    """

    def __init__(self, n_video_features: int = 257, n_audio_features: int = 257, max_duration: int = 40):
        super().__init__()

        self.a_attn_block = ModalMapAttnBlock(n_audio_features, n_video_features, max_duration)
        self.v_attn_block = ModalMapAttnBlock(n_video_features, n_audio_features, max_duration)

        self.av_attn_block = CrossModalMapAttnBlock(n_video_features, max_duration)

    def forward(self, video_feature: Tensor, audio_feature: Tensor, fusion_feature:Tensor, video_bm: Tensor, audio_bm: Tensor, fusion_bm:Tensor) -> Tensor:
        a_attn = self.a_attn_block(audio_bm, audio_feature, video_feature)
        v_attn = self.v_attn_block(video_bm, video_feature, audio_feature)

        sum_attn = a_attn + v_attn 

        a_w = a_attn / sum_attn
        v_w = v_attn / sum_attn

        mid_fusion_bm = video_bm * v_w + audio_bm * a_w

        fusion_attn = self.av_attn_block(fusion_bm, fusion_feature, mid_fusion_bm)

        return fusion_bm * fusion_attn + mid_fusion_bm * (1-fusion_attn)

class ModalMapAttnBlock(nn.Module):

    def __init__(self, n_self_features: int, n_another_features: int, max_duration: int = 40):
        super().__init__()
        self.attn_from_self_features = Conv1d(n_self_features, max_duration, kernel_size=1)
        self.attn_from_another_features = Conv1d(n_another_features, max_duration, kernel_size=1)
        self.attn_from_bm = Conv1d(max_duration, max_duration, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, self_bm: Tensor, self_features: Tensor, another_features: Tensor) -> Tensor:
        w_bm = self.attn_from_bm(self_bm)
        w_self_feat = self.attn_from_self_features(self_features)
        w_another_feat = self.attn_from_another_features(another_features)
        w_stack = torch.stack((w_bm, w_self_feat, w_another_feat), dim=3)
        w = w_stack.mean(dim=3)
        return self.sigmoid(w)

class CrossModalMapAttnBlock(nn.Module):

    def __init__(self, n_self_features: int, max_duration: int = 40):
        super().__init__()
        self.attn_from_self_features = Conv1d(n_self_features, max_duration, kernel_size=1)
        self.attn_from_another_bm = Conv1d(max_duration, max_duration, kernel_size=1)
        self.attn_from_bm = Conv1d(max_duration, max_duration, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, self_bm: Tensor, self_features: Tensor, another_bm: Tensor) -> Tensor:
        w_bm = self.attn_from_bm(self_bm)
        w_self_feat = self.attn_from_self_features(self_features)
        w_another_feat = self.attn_from_another_bm(another_bm)
        w_stack = torch.stack((w_bm, w_self_feat, w_another_feat), dim=3)
        w = w_stack.mean(dim=3)
        return self.sigmoid(w)