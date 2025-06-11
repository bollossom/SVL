import torch
import torch.nn as nn
from timm.layers import to_2tuple, trunc_normal_, DropPath
from collections import OrderedDict
from .misc import *
import time

from .checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

class ReLUX(nn.Module):
    def __init__(self, thre=4):
        super(ReLUX, self).__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)

relu4 = ReLUX(thre=4)

class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=4):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu4(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None
    
class MultiSpike(nn.Module):
    def __init__(self, lens=4, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(inputs)

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        B, N, C = xyz.shape
        if C > 3:
            data = xyz
            xyz = data[:,:,:3]
            rgb = data[:, :, 3:]
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)

        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(batch_size, self.num_group, self.group_size, -1).contiguous()

        # normalize xyz 
        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            neighborhood = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
        else:
            neighborhood = neighborhood_xyz
        return neighborhood, center



class Encoder(nn.Module):
    def __init__(self, encoder_channel, point_input_dims=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.point_input_dims = point_input_dims
        self.first_conv = nn.Sequential(
            nn.Conv2d(self.point_input_dims, 128, 1),
            nn.BatchNorm2d(128),
            MultiSpike(),
            nn.Conv2d(128, 256, 1),            
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            MultiSpike(),
            nn.Conv2d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, c = point_groups.shape
        feature = self.first_conv(point_groups.permute(0,3,1,2))  # B C N G 
        feature_global = torch.max(feature, dim=-1, keepdim=True)[0]  # B C N 1 
        feature = torch.cat([feature_global.expand(-1, -1, -1, n), feature], dim=1)  # B 2*C N G
        feature = self.second_conv(feature)  # B C N G
        feature_global = torch.max(feature, dim=-1, keepdim=False)[0]  # B C N 
        return feature_global.transpose(-1,-2)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.spike = MultiSpike()
        self.bn1 = nn.BatchNorm1d(hidden_features)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        

    def forward(self, x):
        x = self.spike(x)
        x = self.fc1(x)
        x = self.bn1(x.transpose(2,1)).transpose(2,1)
        x= self.spike(x)
        x = self.fc2(x)
        x = self.bn2(x.transpose(2,1)).transpose(2,1)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5
        self.head_spike = MultiSpike()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_norm = nn.BatchNorm1d(dim*3)
        self.q_spike = MultiSpike()
        self.k_spike = MultiSpike()
        self.v_spike = MultiSpike()
        
        self.proj = nn.Linear(dim, dim)
        self.attn_spike = MultiSpike()
        self.project_spike = MultiSpike()

    def forward(self, x):
        B, N, C = x.shape
        x=self.head_spike(x)
        qkv = self.qkv(x)
        qkv = self.qkv_norm(qkv.transpose(2,1)).transpose(2,1)
        qkv= qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q_spike=self.q_spike(q)
        k_spike=self.k_spike(k)
        v_spike=self.v_spike(v)
        attn = q_spike @ k_spike.transpose(-2, -1)
        attn_spike = self.attn_spike(attn)
        x = (attn_spike @ v_spike) * self.scale
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.project_spike(x)
        x = self.proj(x)
        return x
    


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.):
        super().__init__()
        # self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class PointTransformer(nn.Module):
    def __init__(self, config, use_max_pool=True):
        super().__init__()
        self.config = config
        
        self.use_max_pool = config["use_max_pool"] # * whethet to max pool the features of different tokens
        self.trans_dim = config['trans_dim']
        self.depth = config['depth']
        self.drop_path_rate = config['drop_path_rate']
        self.cls_dim = config['cls_dim']
        self.num_heads = config['num_heads']

        self.group_size = config['group_size']
        self.num_group = config['num_group']
        self.point_dims = config['point_dims']
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config['encoder_dims']
        self.encoder = Encoder(encoder_channel=self.encoder_dims, point_input_dims=self.point_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            MultiSpike(),
            nn.Conv1d(128, self.trans_dim, 1),
            nn.BatchNorm1d(self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.BatchNorm1d(self.trans_dim)
        self.project = nn.Linear(768, 1024)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_checkpoint(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if k.startswith('module.'):
                state_dict[k.replace('module.', '')] = v

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                get_missing_parameters_message(incompatible.missing_keys) 
            )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys) 
            )
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            # * print successful loading
            print("PointBERT's weights are successfully loaded from {}".format(bert_ckpt_path))

    def forward(self, xyz, features):
        # divide the point cloud in the same form. This is important
        pts=features.contiguous()
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center.transpose(2,1)).transpose(2,1)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x.transpose(2,1)).transpose(2,1) # * B, G + 1(cls token)(513), C(384)
        if not self.use_max_pool:
            return x
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1).unsqueeze(1)
        # concat_f = self.project(concat_f.squeeze(1)) # * concat the cls token and max pool the features of different tokens, make it B, 1, C
        return concat_f # * B, 1, C(384 + 384)


def make(cfg):
    config = {
        'NAME': 'PointTransformer',
        'trans_dim': 384, 
        'depth': 12, 
        'drop_path_rate': 0.1, 
        'cls_dim': 40, 
        'num_heads': 6,
        'group_size': 32, 
        'num_group': 512,
        'encoder_dims': 256,
        'point_dims': 6,
        'use_max_pool': True
        }
    model_base = PointTransformer(config=config)
    return model_base
