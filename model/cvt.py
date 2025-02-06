from collections.abc import Iterable
from functools import partial
from itertools import repeat
import logging
import os
from collections import OrderedDict
from torchsummary import summary

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einops
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
        # self.attn = Attention(
        #     dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
        #     **kwargs
        # )
        self.attn = EfficientAdditiveAttention(dim_in=dim_in, dim_out=dim_out, num_heads=num_heads)
        # self.attn = AgentAttention(dim_in, num_heads, qkv_bias, attn_drop, drop, agent_num=49, window=20, **kwargs)

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
        # attn = self.attn(x, h, w)
        attn = self.attn(x)
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
                        or pretrained_layers[0] == '*'
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


class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, window=14, with_cls_token=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.with_cls_token = with_cls_token

        self.agent_num = agent_num

        # 窗口大小,等于第71/72行的h和w, 即h=w=window
        self.window = window

        # 深度卷积
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)

        # 池化尺寸 49**0.5=7   agent_num = pool_size*pool_size
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        # agent bias in attention_1
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7,
                                                7))  # block bias (agent_num,h0,w0)   h0,w0是比h,w(h=w=window)小得多的预定义超参数, 因为后面要经过插值恢复h,w
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))  # 行bias (agent_num,h,1)
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))  # 列bias (agent_num,1,w)

        # agent bias in attention_2
        self.na_bias = nn.Parameter(
            torch.zeros(num_heads, agent_num, 7, 7))  # block bias (agent_num,h0,w0)   h0,w0是比h,w小得多的预定义超参数
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))  # 行bias (h,1,agent_num)
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))  # 列bias (1,w,agent_num)

        # 生成std=.02的正态分布数据
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)

    def forward(self, x, h=20, w=20):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, x.shape[1] - 1], 1)

        b, n, c = x.shape
        h = int(n ** 0.5)  # 每列有h个token
        w = int(n ** 0.5)  # 每行有w个token
        num_heads = self.num_heads  # 注意力头的个数
        head_dim = c // num_heads  # 每个头的通道数

        # 将输入x通过线性层生成qkv表示: (b,n,c) --qkv-> (b,n,3c) --reshape-> (b,n,3,c) --permute-> (3,b,n,c)
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        # 将qkv拆分: q:(b,n,c)  k:(b,n,c)  v:(b,n,c)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 对q进行变换,以便进行计算: (b,n,c) --> (b, h, w, c) --> (b,c,h,w)
        q_t = q.reshape(b, h, w, c).permute(0, 3, 1, 2)

        # 通过对q进行池化得到agent表示: (b,c,h,w) --pool-> (b,c, pool_size, pool_size) --reshape-> (b,c,agent_num) --permute-> (b,agent_num,c)
        agent_tokens = self.pool(q_t).reshape(b, c, -1).permute(0, 2, 1)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1,
                                                         3)  # 对q进行变换:(b,n,c) --> (b,n,h,d) --> (b,h,n,d)   c=h*d
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # 对k进行变换:(b,n,c) --> (b,n,h,d) --> (b,h,n,d)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # 对v进行变换:(b,n,c) --> (b,n,h,d) --> (b,h,n,d)

        # 对agent进行变换: (b,agent_num,c) --reshape-> (b,agent_num,h,d) --permute-> (b,h,agent_num,d)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        ### 聚合来自K和V的信息 (第一次注意力)  ###

        # 通过插值将block bias重塑为与输入相同的token数量: (h, agent_num, 7, 7) --> (h, agent_num, window, window);   n=window*window, n:输入token的数量  (初始化的block bias比较小,减少参数量,然后通过插值恢复与输入的相同的token数量)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        # 重塑block bias,变换为与第一次注意力矩阵(shape为:agent_num * token_num)相同的shape: (h, agent_num, window, window) --> (1,h,agent_num,window*window) -->repeat--> (b,h,agent_num,n)
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)

        # 行bias+列bias: (1,h,agent_num,window,1) + (1,h,agent_num,1,window) = (1,h,agent_num,window,window) --reshape-> (1,h,agent_num,n) --repeat--> (b,h,agent_num,n)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)

        # block bias+行bias+列bias: (b,h,agent_num,n) + (b,h,agent_num,n) = (b,h,agent_num,n)
        position_bias = position_bias1 + position_bias2

        # 计算agent和k之间的注意力矩阵: (b,h,agent_num,d) @ (b,h,d,n) = (b,h,agent_num,n);  (b,h,agent_num,n) + (b,h,agent_num,n) = (b,h,agent_num,n)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)

        agent_v = agent_attn @ v  # 聚合value矩阵的信息: (b,h,agent_num,n) @ (b,h,n,d) = (b,h,agent_num,d)

        ### 把信息广播回Q (第二次注意力) ###

        # 通过插值将block bias重塑为与输入相同的token数量: . (h, agent_num, 7, 7) --> (h, agent_num, window, window)   n=window*window  n:token的数量
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        # 重塑block bias,变换为与第二次注意力矩阵(shape为:token_num * agent_num)相同的shape: (h, agent_num, window, window) --reshape-> (1,h,agent_num,window*window) --permute--> (1,h,window*window,agent_num) --repeat-> (b,h,n,agent_num)
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)

        # 行bias+列bias:(1,h,window,1,agent_num) + (1,h,1,window,agent_num) = (1,h,window,window,agent_num) --reshape--> (1,h,window*window,agent_num) -->repeat--> (b,h,n,agent_num)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)

        # block bias+行bias+列bias: (b,h,n,agent_num) + (b,h,n,agent_num) = (b,h,n,agent_num)
        agent_bias = agent_bias1 + agent_bias2

        # 计算q和agent之间的注意力矩阵: (b,h,n,d) @ (b,h,d,agent_num) = (b,h,n,agent_num);   (b,h,n,agent_num) + (b,h,n,agent_num) = (b,h,n,agent_num)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)

        x = q_attn @ agent_v  # 聚合agent_v矩阵的信息: (b,h,n,agent_num)  @ (b,h,agent_num,d) = (b,h,n,d)

        ### 增加深度卷积 ###
        # 对x重塑shape,便于后续计算: (b,h,n,d) --> (b,n,h,d) --> (b,n,c);  c=h*d
        x = x.transpose(1, 2).reshape(b, n, c)

        # 对value矩阵重塑shape,便于进行深度卷积: (b,h,n,d) --transpose--> (b,n,h,d) --reshape--> (b,H,W,c) --permute-> (b,c,H,W);  n=HW
        v_ = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # 增加深度卷积操作: (b,c,H,W)--permute-->(b,H,W,c)-->reshape-->(b,n,c); 并添加残差连接:(b,n,c)+(b,n,c)=(b,n,c)
        x = x + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)

        if self.with_cls_token:
            x = torch.cat((cls_token, x), dim=1)

        x = self.proj(x)  # 输出映射: (b,n,c)-->(b,n,c)
        x = self.proj_drop(x)
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

    def forward(self, x, h=20, w=20):
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


class EfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 ):
        super().__init__()

        self.to_query = nn.Linear(dim_in, dim_out * num_heads)
        self.to_key = nn.Linear(dim_in, dim_out * num_heads)

        self.w_g = nn.Parameter(torch.randn(dim_out * num_heads, 1))
        self.scale_factor = dim_out ** -0.5
        self.Proj = nn.Linear(dim_out * num_heads, dim_out * num_heads)
        self.final = nn.Linear(dim_out * num_heads, dim_out)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD

        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * query, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD

        out = self.Proj(G * key) + query  # BxNxD

        out = self.final(out)  # BxNxD

        return out


if __name__ == '__main__':
    X = torch.randn(1, 401, 256)
    B, N, D = X.size()

    # Model = AgentAttention(dim=256, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
    #                        agent_num=49, window=20, with_cls_token=True)
    # out = Model(X)
    # print(out.shape)
    # Model.to("cuda")
    # summary(Model, (401, 256))

    print("**************************************************")
    Model = EfficientAdditiveAttention(dim_in=256, dim_out=256, num_heads=2)
    out = Model(X)
    print(out.shape)
    Model.to("cuda")
    summary(Model, (401, 256))

    print("**************************************************")
    Model2 = Attention(dim_in=D, dim_out=D, num_heads=4)
    out = Model2(X, 20, 20)
    print(out.shape)
    Model2.to("cuda")
    summary(Model2, (401, 256))
