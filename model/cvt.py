from collections.abc import Iterable
from functools import partial
from itertools import repeat
import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from timm.layers import DropPath, trunc_normal_


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PointwiseConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features, with_cls_token=True):
        super().__init__()
        self.with_cls_token = with_cls_token
        self.net = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_features, in_features, 1),
        )

    def forward(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.net(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.with_cls_token:
            x = torch.cat((cls_token, x), dim=1)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w):
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T - 1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
                hasattr(module, 'conv_proj_q')
                and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
                hasattr(module, 'conv_proj_k')
                and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
                hasattr(module, 'conv_proj_v')
                and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        # self.mlp = Mlp(
        #     in_features=dim_out,
        #     hidden_features=dim_mlp_hidden,
        #     act_layer=act_layer,
        #     drop=drop
        # )
        self.mlp = PointwiseConvMlp(in_features=dim_out, hidden_features=dim_mlp_hidden)

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x), h=h, w=w))

        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                            .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')

        return layers

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


"Fast Vision Transformers with HiLo Attention"

class HiLo(nn.Module):
    """
    HiLo Attention

    Paper: Fast Vision Transformers with HiLo Attention
    Link: https://arxiv.org/abs/2205.13213
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads) # 每个注意力头的通道数
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)  # 根据alpha来确定分配给低频注意力的注意力头的数量
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim   # 确定低频注意力的通道数

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads # 总的注意力头个数-低频注意力头的个数==高频注意力头的个数
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim  # 确定高频注意力的通道数, 总通道数-低频注意力通道数==高频注意力通道数

        # local window size. The `s` in our paper.
        self.ws = window_size  # 窗口的尺寸, 如果ws==2, 那么这个窗口就包含4个patch(或token)

        # 如果窗口的尺寸等于1,这就相当于标准的自注意力机制了, 不存在窗口注意力了; 因此,也就没有高频的操作了,只剩下低频注意力机制了
        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        # 如果低频注意力头的个数大于0, 那就说明存在低频注意力机制。 然后,如果窗口尺寸不为1, 那么应当为每一个窗口应用平均池化操作获得低频信息,这样有助于降低低频注意力机制的计算复杂度 （如果窗口尺寸为1,那么池化层就没有意义了）
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention (Hi-Fi)
        # 如果高频注意力头的个数大于0, 那就说明存在高频注意力机制
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    # 高频注意力机制
    def hifi(self, x):

        B, H, W, C = x.shape

        # 每行有w_group个窗口, 每列有h_group个窗口;
        h_group, w_group = H // self.ws, W // self.ws

        # 总共有total_groups个窗口; 例如：HW=14*14=196个patch; 窗口尺寸为ws=2表示:每行每列有2个patch; 总共有:(14/2)*(14/2)=49个窗口,每个窗口有2*2=4个patch
        total_groups = h_group * w_group

        #通过reshape操作重塑X: (B,H,W,C) --> (B,h_group,ws,w_group,ws,C) --> (B,h_group,w_group,ws,ws,C)   H=h_group*ws, W=w_group*ws
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        # 通过线性层生成qkv: (B,h_group,w_group,ws,ws,C) --> (B,h_group,w_group,ws,ws,3*h_dim) --> (B,total_groups,ws*ws,3,h_heads,head_dim) -->(3,B,total_groups,h_heads,ws*ws,head_dim)    h_dim=h_heads*head_dim
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        # q:(B,total_groups,h_heads,ws*ws,head_dim); k:(B,total_groups,h_heads,ws*ws,head_dim); v:(B,total_groups,h_heads,ws*ws,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 在每个窗口内计算: 所有patch pairs之间的注意力得分: (B,total_groups,h_heads,ws*ws,head_dim) @ (B,total_groups,h_heads,head_dim,ws*ws) = (B,total_groups,h_heads,ws*ws,ws*ws);  ws*ws:表示一个窗口内的patch的数量
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 通过注意力矩阵对Value矩阵进行加权: (B,total_groups,h_heads,ws*ws,ws*ws) @ (B,total_groups,h_heads,ws*ws,head_dim) = (B,total_groups,h_heads,ws*ws,head_dim) --transpose->(B,total_groups,ws*ws,h_heads,head_dim)--reshape-> (B,h_group,w_group,ws,ws,h_dim) ;    h_dim=h_heads*head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)

        # 通过reshape操作重塑, 恢复与输入相同的shape: (B,h_group,w_group,ws,ws,h_dim) --transpose-> (B,h_group,ws,w_group,ws,h_dim) --reshape-> (B,h_group*ws,w_group*ws,h_dim) ==(B,H,W,h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
        # 通过映射层进行输出: (B,H,W,h_dim)--> (B,H,W,h_dim)
        x = self.h_proj(x)

        return x

    # 低频注意力机制
    def lofi(self, x):
        B, H, W, C = x.shape
        # 低频注意力机制中的query来自原始输入x: (B,H,W,C) --> (B,H,W,l_dim) --> (B,HW,l_heads,head_dim) -->(B,l_heads,HW,head_dim);   l_dim=l_heads*head_dim;
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        # 如果窗口尺寸大于1, 在每个窗口执行池化 (如果窗口尺寸等于1,没有池化的必要)
        if self.ws > 1:
            # 重塑维度以便进行池化操作:(B,H,W,C) --> (B,C,H,W)
            x_ = x.permute(0, 3, 1, 2)
            # 在每个窗口执行池化操作: (B,C,H,W) --sr-> (B,C,H/ws,W/ws) --reshape-> (B,C,HW/(ws^2)) --permute-> (B, HW/(ws^2), C);   HW=patch的总数, 每个池化窗口内有: (ws^2)个patch, 池化完还剩下：HW/(ws^2)个patch; 例如：HW=196个patch,每个池化窗口有(2^2=4)个patch,池化完还剩下49个patch【每个patch汇总了之前4个patch的信息】
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # 将池化后的输出通过线性层生成kv:(B,HW/(ws^2),C) --l_kv-> (B,HW/(ws^2),l_dim*2) --reshape-> (B,HW/(ws^2),2,l_heads,head_dim) --permute-> (2,B,l_heads,HW/(ws^2),head_dim)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            # 如果窗口尺寸等于1, 那么kv和q一样, 来源于原始输入x: (B,H,W,C) --l_kv-> (B,H,W,l_dim*2) --reshape-> (B,HW,2,l_heads,head_dim) --permute-> (2,B,l_heads,HW,head_dim);  【注意: 如果窗口尺寸为1,那就不会执行池化操作,所以patch的数量也不会减少,依然是HW个patch】
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)

        # 以ws>1为例: k:(B,l_heads,HW/(ws^2),head_dim);  v:(B,l_heads,HW/(ws^2),head_dim)
        k, v = kv[0], kv[1]

        # 计算q和k之间的注意力矩阵: (B,l_heads,HW,head_dim) @ (B,l_heads,head_dim,HW/(ws^2)) == (B,l_heads,HW,HW/(ws^2))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 通过注意力矩阵对Value矩阵进行加权: (B,l_heads,HW,HW/(ws^2)) @ (B,l_heads,HW/(ws^2),head_dim) == (B,l_heads,HW,head_dim) --transpose->(B,HW,l_heads,head_dim)--reshape-> (B,H,W,l_dim);   l_dim=l_heads*head_dim
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        # 通过映射层输出: (B,H,W,l_dim)-->(B,H,W,l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x):
        res = self.norm(x)
        x = res
        B, N, C = x.shape
        # H = W = 每一列/行有多少个patch
        H = W = int(N ** 0.5)
        # 将X重塑为四维: (B,N,C) --> (B,H,W,C)   【注意: 这里的H/W并不是图像的高和宽】
        x = x.reshape(B, H, W, C)

        # 如果分配给高频注意力的注意力头的个数为0,那么仅仅执行低频注意力
        if self.h_heads == 0:
            # (B,H,W,C) --> (B,H,W,l_dim)  此时,C=l_dim,因为所有的注意力头都分配给了低频注意力
            x = self.lofi(x)
            return x.reshape(B, N, C)

        # 如果分配给低频注意力的注意力头的个数为0,那么仅仅执行高频注意力
        if self.l_heads == 0:
            # 执行高频注意力: (B,H,W,C) --> (B,H,W,h_dim); 此时,C=h_dim,因为所有的注意力头都分配给了高频注意力
            x = self.hifi(x)
            return x.reshape(B, N, C)

        # 执行高频注意力: (B,H,W,C) --> (B,H,W,h_dim)
        hifi_out = self.hifi(x)
        # 执行低频注意力: (B,H,W,C) --> (B,H,W,l_dim)
        lofi_out = self.lofi(x)

        # 在通道方向上拼接高频注意力和低频注意力的输出: (B,H,W,h_dim+l_dim)== (B,H,W,C)
        x = torch.cat((hifi_out, lofi_out), dim=-1)
        # 将输出重塑为与输入相同的shape: (B,H,W,C)-->(B,N,C)
        x = x.reshape(B, N, C)

        return res + x


