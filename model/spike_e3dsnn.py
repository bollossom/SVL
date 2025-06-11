from functools import partial
import torch
import torch.nn as nn
from einops import rearrange
import spconv.pytorch as spconv
from torch_geometric.utils import scatter
from timm.layers import trunc_normal_
from utils.misc import offset2batch


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
    
class Multispike(nn.Module):
    def __init__(self, lens=4, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(inputs)
    
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_fn=None,
        indice_key=None,
        depth=4,
        groups=None,
        grid_size=None,
        bias=False,
    ):
        super().__init__()
        assert embed_channels % groups == 0
        self.groups = groups
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.block = spconv.SparseSequential(
            Multispike(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels))
        self.voxel_block = spconv.SparseSequential(
            Multispike(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
        )

    def forward(self, x):
        feat = x
        feat = self.block(x) + x.features
        res = feat
        x = feat
        x = self.voxel_block(x)
        x = x.replace_feature(x.features + res.features)
        return x


class DonwBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        depth,
        sp_indice_key,
        point_grid_size,
        num_ref=16,
        groups=None,
        norm_fn=None,
        sub_indice_key=None,
    ):
        super().__init__()
        self.num_ref = num_ref
        self.depth = depth
        self.point_grid_size = point_grid_size
        self.down = spconv.SparseSequential(
            Multispike(),
            spconv.SparseConv3d(
                in_channels,
                embed_channels,
                kernel_size=2,
                stride=2,
                indice_key=sp_indice_key,
                bias=False,
            ),
            norm_fn(embed_channels),
        )
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BasicBlock(
                    in_channels=embed_channels,
                    embed_channels=embed_channels,
                    depth=len(point_grid_size) + 1,
                    groups=groups,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                )
            )

    def forward(self, x):
        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        return x

class E_3DSNN_T(nn.Module):
    def __init__(
        self,
        config,
        embed_channels=16,
        enc_num_ref=[16, 16, 16, 16],
        enc_channels=[16, 32, 64, 128],
        groups=[2, 4, 8, 16],
        enc_depth=[1, 1, 1, 1],
        point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]],        
    ):
        super().__init__()
        self.in_channels = config.model.in_channel
        self.num_classes = config.model.out_channel
        self.voxel_size = config.model.voxel_size
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.in_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
        )

        self.enc = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                DonwBlock(
                    in_channels=self.embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=point_grid_size[i],
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                ))

        final_in_channels = enc_channels[-1]
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, self.num_classes, kernel_size=1, padding=1, bias=True
            )
            if self.num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    def forward(self, xyz, feat):
        offset = torch.cumsum(torch.tensor([x.shape[0] for x in xyz]), dim=0)
        grid_coord = torch.div(xyz - xyz.min(dim=0, keepdim=True)[0], self.voxel_size, rounding_mode="trunc").int()
        batch = offset2batch(offset)

        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1), grid_coord], dim=1)
            .int()
            .contiguous(),
            spatial_shape=torch.add(
                torch.max(grid_coord, dim=0).values, 96
            ).tolist(),
            batch_size=batch[-1].tolist() + 1,
        )

        x = self.stem(x)

        for i in range(self.num_stages):
            x = self.enc[i](x)

        x = self.final(x)

        x = x.replace_feature(
                 scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
             )
        return x.features

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class E_3DSNN_S(nn.Module):
    def __init__(
        self,
        config,
        embed_channels=24,
        enc_num_ref=[16, 16, 16, 16],
        enc_channels=[24, 48, 96, 160],
        groups=[2, 4, 8, 16],
        enc_depth=[1, 1, 1, 1],
        point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]],        
    ):
        super().__init__()
        self.in_channels = config.model.in_channel
        self.num_classes = config.model.out_channel
        self.voxel_size = config.model.voxel_size
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.in_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
        )

        self.enc = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                DonwBlock(
                    in_channels=self.embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=point_grid_size[i],
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                ))

        final_in_channels = enc_channels[-1]
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, self.num_classes, kernel_size=1, padding=1, bias=True
            )
            if self.num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    def forward(self, xyz, feat):
        offset = torch.cumsum(torch.tensor([x.shape[0] for x in xyz]), dim=0)
        grid_coord = torch.div(xyz - xyz.min(dim=0, keepdim=True)[0], self.voxel_size, rounding_mode="trunc").int()
        batch = offset2batch(offset)

        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1), grid_coord], dim=1)
            .int()
            .contiguous(),
            spatial_shape=torch.add(
                torch.max(grid_coord, dim=0).values, 96
            ).tolist(),
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.stem(x)

        for i in range(self.num_stages):
            x = self.enc[i](x)

        x = self.final(x)

        x = x.replace_feature(
                 scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
             )
        return x.features

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class E_3DSNN_L(nn.Module):
    def __init__(
        self,
        config,
        embed_channels=64,
        enc_num_ref=[16, 16, 16, 16],
        enc_channels=[64,128,128,256],
        groups=[2, 4, 8, 16],
        enc_depth=[2,2,2,2],
        point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]],        
    ):
        super().__init__()
        self.in_channels = config.model.in_channel
        self.num_classes = config.model.out_channel
        self.voxel_size = config.model.voxel_size
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.in_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
        )

        self.enc = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                DonwBlock(
                    in_channels=self.embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=point_grid_size[i],
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                ))

        final_in_channels = enc_channels[-1]
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, self.num_classes, kernel_size=1, padding=1, bias=True
            )
            if self.num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    def forward(self, xyz, feat):
        offset = torch.cumsum(torch.tensor([x.shape[0] for x in xyz]), dim=0)
        grid_coord = torch.div(xyz - xyz.min(dim=0, keepdim=True)[0], self.voxel_size, rounding_mode="trunc").int()
        batch = offset2batch(offset)

        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1), grid_coord], dim=1)
            .int()
            .contiguous(),
            spatial_shape=torch.add(
                torch.max(grid_coord, dim=0).values, 96
            ).tolist(),
            batch_size=batch[-1].tolist() + 1,
        )

        x = self.stem(x)

        for i in range(self.num_stages):
            x = self.enc[i](x)

        x = self.final(x)

        x = x.replace_feature(
                 scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
             )
        return x.features

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class E_3DSNN_H(nn.Module):
    def __init__(
        self,
        config,
        embed_channels=64,
        enc_num_ref=[16, 16, 16, 16],
        enc_channels=[96,192,288,384],
        groups=[2, 4, 8, 16],
        enc_depth=[2,2,2,2],
        point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [2, 4, 6]],        
    ):
        super().__init__()
        self.in_channels = config.model.in_channel
        self.num_classes = config.model.out_channel
        self.voxel_size = config.model.voxel_size
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.in_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
            Multispike(),
            spconv.SubMConv3d(
                self.embed_channels,
                self.embed_channels,
                kernel_size=3,
                padding=1,
                indice_key="stem",
                bias=False,
            ),
            norm_fn(self.embed_channels),
        )

        self.enc = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                DonwBlock(
                    in_channels=self.embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=point_grid_size[i],
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                ))

        final_in_channels = enc_channels[-1]
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, self.num_classes, kernel_size=1, padding=1, bias=True
            )
            if self.num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    def forward(self, xyz, feat):
        offset = torch.cumsum(torch.tensor([x.shape[0] for x in xyz]), dim=0)
        grid_coord = torch.div(xyz - xyz.min(dim=0, keepdim=True)[0], self.voxel_size, rounding_mode="trunc").int()
        batch = offset2batch(offset)

        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1), grid_coord], dim=1)
            .int()
            .contiguous(),
            spatial_shape=torch.add(
                torch.max(grid_coord, dim=0).values, 96
            ).tolist(),
            batch_size=batch[-1].tolist() + 1,
        )

        x = self.stem(x)

        for i in range(self.num_stages):
            x = self.enc[i](x)

        x = self.final(x)

        x = x.replace_feature(
                 scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
             )
        return x.features

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
