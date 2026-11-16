import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from net.pvtv2 import pvt_v2_b2
from transformers import BertTokenizer, BertConfig, BertModel, get_linear_schedule_with_warmup
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from random import random



def get_text_feature(x, k=20, device = 'cuda', idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)

    # x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device(device)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]


def get_graph_feature(x, k=20, device = 'cuda', idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

def get_correlation_map(x, prototype_block):
    B, C, H, W = x.size()

    # prototype_block: (B,N,C)
    # x: (B,C,H,W)
    # corr: (B,N,H,W), -1~1
    n_p = prototype_block / prototype_block.norm(dim=2, keepdim=True)
    n_x = x.view(B, C, -1) / x.view(B, C, -1).norm(dim=1, keepdim=True)
    corr = torch.bmm(n_p, n_x).view(B, -1, H, W)
    return corr

def get_correlation_map_text(x, prototype_block):
    B, C, N_l = x.size()

    # prototype_block: (B,N,C)
    # x: (B,C,H,W)
    # corr: (B,N,H,W), -1~1
    n_p = prototype_block / prototype_block.norm(dim=2, keepdim=True)
    n_x = x / x.norm(dim=1, keepdim=True)
    corr = torch.bmm(n_p, n_x).view(B, -1, N_l)
    return corr

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

class ImageTextFusionAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, feature_dim, k_channels, which_conv=nn.Conv2d, num_heads=1, device='cuda'):
        super(ImageTextFusionAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.device = device
        self.v_in_channels = v_in_channels
        self.feature_dim = feature_dim
        self.l_in_channels = l_in_channels
        self.k_channels = k_channels
        self.v_channels = k_channels
        self.num_heads = num_heads
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.v_in_channels, self.feature_dim, kernel_size=1, padding=0, bias=False)

        self.phi = self.which_conv(
            self.v_in_channels, self.feature_dim, kernel_size=1, padding=0, bias=False)

        self.phi_k = [self.which_conv(
            self.v_in_channels, self.feature_dim, kernel_size=1, padding=0,
            bias=False).cuda()]  # using list to prevent pytorch consider phi_k as a parameter to optimize

        for param_phi, param_phi_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
            param_phi_k.data.copy_(param_phi.data)  # initialize
            param_phi_k.requires_grad = False  # not update by gradient

        self.g = self.which_conv(
            self.v_in_channels, self.feature_dim, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.feature_dim, self.v_in_channels, kernel_size=1, padding=0, bias=False)
        # text
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.k_channels, kernel_size=1, stride=1),
        ) # (B, key_channels, N_l)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.v_channels, kernel_size=1, stride=1),
        ) # (B, value_channels, N_l)
    def forward(self, x, l, l_mask, device="cuda"):
        theta = self.theta(x)  # [n, c, h, w]
        phi = self.phi(x)  # [n, c, h, w]
        g = self.g(x)  # [n, c, h, w]

        B, c, h, w = theta.shape
        HW = h * w

        # normal self attention
        theta = theta.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
        g = g.view(-1, self.feature_dim, x.shape[2] * x.shape[3])

        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2).contiguous(), phi), -1)

        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2).contiguous()).view(-1,
                                                                        self.feature_dim, x.shape[2], x.shape[3]))
        o = o.view(B, c, HW)
        query = g.permute(0, 2, 1)  # (B, H*W, c)
        key = self.f_key(l) # (B, key_channels, N_l)
        value = self.f_value(l) # (B, value_channels, N_l)
        l_mask = l_mask.permute(0, 2, 1)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.k_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.k_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.v_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.k_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.v_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        
        return o+out
    

class ImageTextFusionAttention_A(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, feature_dim, k_channels, which_conv=nn.Conv2d, num_heads=1, device='cuda'):
        super(ImageTextFusionAttention_A, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.device = device
        self.v_in_channels = v_in_channels
        self.feature_dim = feature_dim
        self.l_in_channels = l_in_channels
        self.k_channels = k_channels
        self.v_channels = k_channels
        self.num_heads = num_heads
        self.which_conv = which_conv
        self.theta = self.which_conv(
            self.v_in_channels, self.feature_dim, kernel_size=1, padding=0, bias=False)

        self.phi = self.which_conv(
            self.v_in_channels, self.feature_dim, kernel_size=1, padding=0, bias=False)

        self.phi_k = [self.which_conv(
            self.v_in_channels, self.feature_dim, kernel_size=1, padding=0,
            bias=False).cuda()]  # using list to prevent pytorch consider phi_k as a parameter to optimize

        for param_phi, param_phi_k in zip(self.phi.parameters(), self.phi_k[0].parameters()):
            param_phi_k.data.copy_(param_phi.data)  # initialize
            param_phi_k.requires_grad = False  # not update by gradient

        self.g = self.which_conv(
            self.v_in_channels, self.feature_dim, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(
            self.feature_dim, self.v_in_channels, kernel_size=1, padding=0, bias=False)
        # text
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.k_channels, kernel_size=1, stride=1),
        ) # (B, key_channels, N_l)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.v_channels, kernel_size=1, stride=1),
        ) # (B, value_channels, N_l)

        corre_matrix_1 = torch.rand((60, 676)).to(device)
        self.correlation_matrix_1 = nn.Parameter(corre_matrix_1, requires_grad=True)

        corre_matrix_2 = torch.rand((60, 60)).to(device)
        self.correlation_matrix_2 = nn.Parameter(corre_matrix_2, requires_grad=True)

    def forward(self, x, l, l_mask, device="cuda"):
        """
        x shape: (B, 64, 32, 32)
        l shape: (B, 768, 60)
        """
        theta = self.theta(x)  # [n, c, h, w]
        phi = self.phi(x)  # [n, c, h, w]
        g = self.g(x)  # [n, c, h, w]

        B, c, h, w = theta.shape
        HW = h * w

        # normal self attention
        theta = theta.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.feature_dim, x.shape[2] * x.shape[3])
        g = g.view(-1, self.feature_dim, x.shape[2] * x.shape[3])

        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2).contiguous(), phi), -1) # (B, HW, HW)
        
        query = g.permute(0, 2, 1)  # (B, H*W, c)

        key = self.f_key(l) # (B, key_channels, N_l)
        value = self.f_value(l) # (B, value_channels, N_l)
        l_mask = l_mask.permute(0, 2, 1)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.k_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.k_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.v_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.k_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)

        cor_score = torch.bmm(beta, sim_map.squeeze(1))  # (B, HW, N_l)
        beta_weight = beta + torch.bmm(cor_score, self.correlation_matrix_1.unsqueeze(0).expand(B,60,676))
        sim_map_weight = sim_map + torch.bmm(cor_score, self.correlation_matrix_2.unsqueeze(0).expand(B,60,60)).unsqueeze(1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta_weight.transpose(1, 2).contiguous()).view(-1,
                                                                        self.feature_dim, x.shape[2], x.shape[3]))
        o = o.view(B, c, HW)
        out = torch.matmul(sim_map_weight, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.v_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        
        return o+out


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)
        return out

class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=0, dropout=0.0):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )
        self.img_text_fusion = ImageTextFusionAttention_A(v_in_channels, l_in_channels, dim,key_channels)
        # self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            # l_in_channels,  # l_in
                                                            # key_channels,  # key
                                                            # value_channels,  # value
                                                            # out_channels=value_channels,  # out
                                                            # num_heads=num_heads)

        # self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
        #                                 nn.GELU(),
        #                                 nn.Dropout(dropout)
        #                                 )
        self.softmax = nn.Softmax(dim=-1)
        self.img_GAN = Graph_Attention_Network(v_in_channels)
        self.rgb_fusion = nn.Sequential(nn.Conv2d(v_in_channels + v_in_channels, v_in_channels, 1),
                                       nn.BatchNorm2d(v_in_channels),
                                       nn.ReLU(inplace=True),
                                       )
        self.text_GAN = Text_Attention_Network(l_in_channels)
        self.text_fusion = nn.Sequential(nn.Conv1d(l_in_channels + l_in_channels, l_in_channels, 1),
                                       nn.BatchNorm1d(l_in_channels),
                                       nn.ReLU(inplace=True),
                                       )
        self.ambiguity = AmbiguityLearning(dim)


    def augmente_img_feature(self, x):
        # input x shape: (B, dim, H, W, )
        B, dim, H, W = x.shape
        x_ = x.view(B, dim, -1)  # (B, dim, H*W)
        # print(f'x_ shape {x_.shape}')
        ss_map = self.softmax(x_)
        pb = torch.bmm(ss_map, x_.transpose(1, 2))
        # print(f'pb shape {pb.shape}')
        graph_pb = self.img_GAN(pb)
        # print(f'graph_pb shape {graph_pb.shape}')
        cm = get_correlation_map(x, graph_pb)
        # print(f'cm shape {cm.shape}')
        img = self.rgb_fusion(torch.cat([x, cm], dim=1)) # (B, dim, H, W)
        return img
    
    def augment_text_feature(self, l):
        # print('-----------------------------')
        # print(f'ori text shape {l.shape}')
        # l = l.permute(0, 2, 1)  # (B, dim, N_l)
        ss_map = self.softmax(l)
        pb = torch.bmm(ss_map, l.transpose(1, 2))
        # print(f'pb shape {pb.shape}')
        lang_pb = self.text_GAN(pb)
        # print(f'lang_pb shape {lang_pb.shape}')
        cm = get_correlation_map_text(l, lang_pb)
        # print(f'cm shape {cm.shape}')
        lang = self.text_fusion(torch.cat([l, cm], dim=1))
        return lang

    def forward(self, x, l, l_mask):
        # input x shape: (B, dim, H, W)
        # input l shape: (B, dim, N_l)
        # print(l.shape)
        # print(l_mask.shape)
        x_prototype = self.augmente_img_feature(x)
        # x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)  # (B, H*W, dim)
        l_prototype = self.augment_text_feature(l)
        ambiguity_score = self.ambiguity(x_prototype, l_prototype)
        l = (1-ambiguity_score)[:, None, None] * l
        mm = self.img_text_fusion(x,l,l_mask)
        # vis = self.vis_project(x.permute(0, 2, 1))  # (B, dim, H*W)
        # lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)
        # lang = lang.permute(0, 2, 1)  # (B, dim, H*W)
        # lang = (1-ambiguity_score)[:, None, None] * lang
        # mm = torch.mul(vis, lang)
        # mm = self.project_mm(mm)  # (B, dim, H*W)
        # mm = mm.permute(0, 2, 1)  # (B, H*W, dim)
        return mm

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


# class EFM(nn.Module):
#     def __init__(self, hchannel, channel):
#         super(EFM, self).__init__()
#         self.conv1_1 = Conv1x1(hchannel, channel)
#         self.conv1_2 = Conv1x1(channel*4, channel)
#         self.conv3_1 = ConvBNR(channel, channel, 3)
 
#         self.branch0 = nn.Sequential(
#             Conv1x1(hchannel, channel),
#             ConvBNR(channel, channel, 3)
#         )

#         self.branch1 = nn.Sequential(
#             Conv1x1(hchannel, channel),
#             ConvBNR(channel, channel, 3, dilation=2)
#         )
#         self.branch2 = nn.Sequential(
#             Conv1x1(hchannel, channel),
#             ConvBNR(channel, channel, 3, dilation=3)
#         )
#         self.branch3 = nn.Sequential(
#             Conv1x1(hchannel, channel),
#             ConvBNR(channel, channel, 3, dilation=4)
#         )
        
        

#     def forward(self, x):

#         x_local = self.conv1_1(x)

#         x_global0 = self.branch0(x)
#         x_global1 = self.branch1(x) + x_global0
#         x_global2 = self.branch2(x) + x_global1
#         x_global3 = self.branch3(x) + x_global2

#         cat = self.conv1_2(torch.cat((x_global0, x_global1, x_global2, x_global3),1))

#         out = self.conv3_1(cat + x_local)

#         return out


class Transformer(nn.Module):
    def __init__(self, in_channels):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.theta = nn.Linear(self.in_channels, self.inter_channels)
        self.phi = nn.Linear(self.in_channels, self.inter_channels)
        self.g = nn.Linear(self.in_channels, self.inter_channels)
        self.W = nn.Linear(self.inter_channels, self.in_channels)

    def forward(self, ori_feature):
        ori_feature = ori_feature.permute(0, 2, 1)
        feature = self.bn_relu(ori_feature)
        feature = feature.permute(0, 2, 1)
        B, N, C = feature.size()

        x_theta = self.theta(feature)
        x_phi = self.phi(feature)
        x_phi = x_phi.permute(0, 2, 1)
        attention = torch.matmul(x_theta, x_phi)

        f_div_C = F.softmax(attention, dim=-1)
        g_x = self.g(feature)
        y = torch.matmul(f_div_C, g_x)
        W_y = self.W(y).contiguous().view(B, N, C)
        att_fea = ori_feature.permute(0, 2, 1) + W_y
        return att_fea

class Text_Attention_Network(nn.Module):
    def __init__(self, in_channels):
        super(Text_Attention_Network, self).__init__()
        self.TF = Transformer(in_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, prototype_block):
        att_prototype_block = self.TF(prototype_block)
        prototype_for_graph = att_prototype_block.permute(0, 2, 1)
        graph_prototype = get_graph_feature(prototype_for_graph, k=10, device = prototype_for_graph.device)
        graph_prototype = self.conv1(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10, device = prototype_for_graph.device)
        graph_prototype = self.conv2(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10, device = prototype_for_graph.device)
        graph_prototype = self.conv3(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]
        graph_prototype_block = graph_prototype.permute(0, 2, 1)
        return graph_prototype_block

class Graph_Attention_Network(nn.Module):
    def __init__(self, in_channels):
        super(Graph_Attention_Network, self).__init__()
        self.TF = Transformer(in_channels)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, prototype_block):
        att_prototype_block = self.TF(prototype_block)
        prototype_for_graph = att_prototype_block.permute(0, 2, 1)
        graph_prototype = get_graph_feature(prototype_for_graph, k=10)
        graph_prototype = self.conv1(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv2(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]

        graph_prototype = get_graph_feature(graph_prototype, k=10)
        graph_prototype = self.conv3(graph_prototype)
        graph_prototype = graph_prototype.max(dim=-1, keepdim=False)[0]
        graph_prototype_block = graph_prototype.permute(0, 2, 1)
        return graph_prototype_block

    
class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
            nn.BatchNorm1d(z_dim * 2)
        )

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  
        return Independent(Normal(loc=mu, scale=sigma), 1)

class AmbiguityLearning(nn.Module):
    def __init__(self, v_dim):
        super(AmbiguityLearning, self).__init__()
        self.img_process = nn.Sequential(
            nn.Conv2d(v_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.text_process = nn.Sequential(
            nn.Linear(768, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, image_encoding, text_encoding):
        text_encoding = text_encoding.permute(0, 2, 1)
        text_encoding = text_encoding.max(dim=1)[0]
        text_encoding = self.text_process(text_encoding)
        image_encoding = self.img_process(image_encoding)
        image_encoding = image_encoding.flatten(2).max(dim=2)[0]
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1)/ 2.
        skl = nn.functional.sigmoid(skl)
        return skl
    
class Conv_Block(nn.Module):  # [64, 128, 320, 512]
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(512+320+128, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)
        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

class EFM(nn.Module):
    def __init__(self, hchannel, channel):
        super(EFM, self).__init__()
        self.conv1_1 = Conv1x1(hchannel, channel)
        self.conv1_2 = Conv1x1(channel*4, channel)
        self.conv3_1 = ConvBNR(channel, channel, 3)
 
        self.branch0 = nn.Sequential(
            Conv1x1(hchannel, channel),
            ConvBNR(channel, channel, 3)
        )

        self.branch1 = nn.Sequential(
            Conv1x1(hchannel, channel),
            ConvBNR(channel, channel, 3, dilation=2)
        )
        self.branch2 = nn.Sequential(
            Conv1x1(hchannel, channel),
            ConvBNR(channel, channel, 3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            Conv1x1(hchannel, channel),
            ConvBNR(channel, channel, 3, dilation=4)
        )
        
        

    def forward(self, x):

        x_local = self.conv1_1(x)

        x_global0 = self.branch0(x)
        x_global1 = self.branch1(x) + x_global0
        x_global2 = self.branch2(x) + x_global1
        x_global3 = self.branch3(x) + x_global2

        cat = self.conv1_2(torch.cat((x_global0, x_global1, x_global2, x_global3),1))

        out = self.conv3_1(cat + x_local)

        return out

class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.cov1_1 = nn.Conv2d(2*channel, channel, 1)
        self.CBR1 = ConvBNR(channel, channel, 3)
        self.CBR2 = ConvBNR(channel, channel, 3)
        self.CBR3 = ConvBNR(channel, channel, 3)
        self.CBR4 = ConvBNR(channel, channel, 3)
        self.cov1 = nn.Conv2d(2*channel, channel, 1)
        self.ecov1 = ConvBNR(channel // 2, channel // 2, 3)
        self.ecov2 = ConvBNR(channel // 2, channel // 2, 3)
        self.ecov3 = ConvBNR(channel // 2, channel // 2, 3)
        self.conv1_2 = ConvBNR(channel, channel, 3)
        self.conv1_3 = nn.Conv2d(channel, 1, 1)
        


    def forward(self, x_low, x_high):
        if x_low.size() != x_high.size():
            x_high = F.interpolate(x_high, x_low.size()[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat((x_low, x_high), dim=1)
        
        e = self.cov1(x_cat)
        ec = torch.chunk(e, 2, dim=1)
        e1 = self.ecov1(ec[0]) + ec[1]
        e2 = self.ecov2(e1) * ec[0]
        e3 = self.ecov3(e1) + e2

        ee = self.conv1_2(torch.cat((e1, e3), dim=1))
        edge_att = torch.sigmoid(self.conv1_3(ee))

        x11 = self.cov1_1(x_cat)
        x1 = self.CBR1(x11)
        x2 = self.CBR2(x1)
        x3 = self.CBR3(x2 + x11)
        edge = x3 * edge_att
        x4 = self.CBR4(x3 + edge)
        return x4, edge_att



#Laddered Converged Network
class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()

        self.backbone = pvt_v2_b2()  
        # self.config = BertConfig.from_pretrained(CASED)
        self.bert = BertModel.from_pretrained(opt.bert_path)
        path = opt.pvt_checkpoint
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        # fusion
        # self.v_channels = [64, 128, 320, 512, 64]
        # self.fusion_layers = nn.ModuleList()
        # for i in range(5):
        #     layer = PWAM(self.v_channels[i], self.v_channels[i], 768, self.v_channels[i], self.v_channels[i], 1, dropout=0.5)
        #     self.fusion_layers.append(layer)
        self.fusion_layer = PWAM(64, 64, 768, 64, 64, 1, dropout=0.5)

        self.ef1 = EFM(64, 64)
        self.ef2 = EFM(128, 64)
        self.ef3 = EFM(320, 64)
        self.ef4 = EFM(512, 64)
        self.ef5 = EFM(64, 64)

        # layer fusion
        self.fusion45 = Decoder(64)
        self.fusion34 = Decoder(64)
        self.fusion23 = Decoder(64)
        self.fusion12 = Decoder(64)

        self.conv_block = Conv_Block(64)


        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(64, 1, 1)
        self.predictor3 = nn.Conv2d(64, 1, 1)
        self.predictor4 = nn.Conv2d(64, 1, 1)


    def forward(self, x, text_id, text_mask, text_type_id):
        image_shape = x.size()[2:]
        text_features = self.bert(input_ids=text_id,attention_mask=text_mask,token_type_ids=text_type_id)[0]

        pvt = self.backbone(x) # [64, 128, 320, 512, 512] [88,88  44,44  22,22 11,11, 22,22]
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        # print(x1.shape, x2.shape, x3.shape, x4.shape)

        if x4.size()[2:] != x3.size()[2:]:
            x41 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear')
        if x2.size()[2:] != x3.size()[2:]:
            x21 = F.interpolate(x2, size=x3.size()[2:], mode='bilinear')
        x5 = self.conv_block(x41, x3, x21)

        text_output = text_features.permute(0, 2, 1)
        lang_mask = text_mask.unsqueeze(dim=-1)
        
        # ti1 = self.fusion_layers[0](x1, text_output, lang_mask) # torch.Size([B, 3136, 64])
        # ti2 = self.fusion_layers[1](x2, text_output, lang_mask) # torch.Size([B, 784, 128])
        # ti3 = self.fusion_layers[2](x3, text_output, lang_mask) # torch.Size([B, 196, 320])
        # ti4 = self.fusion_layers[3](x4, text_output, lang_mask) # torch.Size([B, 49, 512])
        ti5 = self.fusion_layer(x5, text_output, lang_mask) # torch.Size([B, 32*32, 64])
        


        x1_fusion = self.ef1(x1) # torch.Size([B, 64, 56, 56])
        x2_fusion = self.ef2(x2) # torch.Size([B, 64, 28, 28])
        x3_fusion = self.ef3(x3) # torch.Size([B, 64, 14, 14])
        x4_fusion = self.ef4(x4) # torch.Size([3B, 64, 7, 7])
        x5_fusion = self.ef5(ti5.view(-1, 26, 26, 64).permute(0, 3, 1, 2)) # torch.Size([B, 64, 7, 7])
        
        # layer fusion
        x45, edge45 = self.fusion45(x4_fusion, x5_fusion) # torch.Size([B, 64, 14, 14])
        x345, edge345 = self.fusion34(x3_fusion, x45) # torch.Size([B, 64, 14, 14])
        x2345, edge2345 = self.fusion23(x2_fusion, x345) # torch.Size([B, 64, 28, 28])
        x12345, edge12345 = self.fusion12(x1_fusion, x2345) # torch.Size([B, 64, 56, 56])

        o3 = self.predictor4(x45)
        o3 = F.interpolate(o3, size=image_shape, mode='bilinear', align_corners=False)
        o2 = self.predictor3(x345)
        o2 = F.interpolate(o2, size=image_shape, mode='bilinear', align_corners=False)
        o1 = self.predictor2(x2345)
        o1 = F.interpolate(o1, size=image_shape, mode='bilinear', align_corners=False)
        o0 = self.predictor1(x12345)
        o0 = F.interpolate(o0, size=image_shape, mode='bilinear', align_corners=False)

        oe45 = F.interpolate(edge45, size=image_shape, mode='bilinear', align_corners=False)
        oe345 = F.interpolate(edge345, size=image_shape, mode='bilinear', align_corners=False)
        oe2345 = F.interpolate(edge2345, size=image_shape, mode='bilinear', align_corners=False)
        oe12345 = F.interpolate(edge12345, size=image_shape, mode='bilinear', align_corners=False)

        return o0, o1, o2, o3, oe45, oe345, oe2345, oe12345


def load_checkpoint(model, path):
    state_dict = torch.load(path, map_location='cpu')
    nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, 'module.')
    model.load_state_dict(state_dict, strict=True)
