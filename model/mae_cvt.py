import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from model.cvt import ConvEmbed, Block


class MaskedAutoencoderCvT(nn.Module):
    def __init__(self, img_size=(512, 512), patch_size=16, in_chans=3, out_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_only_masked_tokens_ab=False, abnormal_score_func='L1',
                 grad_weighted_loss=True, is_target=False):
        super().__init__()
        # --------------------------------------------------------------------------
        # Abnormal specifics
        self.H = 0
        self.W = 0
        self.masked_H = 0
        self.masked_W = 0
        self.use_only_masked_tokens_ab = use_only_masked_tokens_ab
        self.abnormal_score_func = abnormal_score_func
        self.grad_weighted_loss = grad_weighted_loss
        self.norm_pix_loss = norm_pix_loss
        self.is_target = is_target

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_size,
            padding=0,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.patch_size = patch_size
        self.num_patches = img_size[0]//patch_size*img_size[1]//patch_size
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)  # decoder to patch
        self.out_chans = out_chans
        # --------------------------------------------------------------------------

    def patchify(self, img):
        """
        img: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert img.shape[2] % p == 0 and img.shape[3] % p == 0

        h = img.shape[2] // p
        w = img.shape[3] // p

        x = img.reshape(shape=(img.shape[0], self.out_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(img.shape[0], h * w, p ** 2 * self.out_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        img: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = 20
        w = 40
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        img = x.reshape(shape=(x.shape[0], self.out_chans, h * p, w * p))
        return img

    def random_masking(self, x, mask_ratio, ids_shuffle=None, ids_restore=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, D, H, W = x.shape  # batch, length, dim
        L = H*W
        x = rearrange(x, 'b c h w -> b (h w) c')
        len_keep = int(L * (1 - mask_ratio))

        if ids_shuffle is None or ids_restore is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.masked_H = H
        self.masked_W = int(W*(1.-mask_ratio))
        self.H = H
        self.W = W
        # x_masked = rearrange(x_masked, 'b (h w) c -> b c h w', h=self.masked_H, w=self.masked_W)
        return x_masked, mask, ids_shuffle, ids_restore

    def forward_encoder(self, x, mask_ratio, ids_shuffle=None, ids_restore=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_shuffle, ids_restore = self.random_masking(x, mask_ratio, ids_shuffle, ids_restore)
        # x = rearrange(x, 'b c h w -> b (h w) c')
        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, self.masked_H, self.masked_W)
        x = self.norm(x)

        return x, mask, ids_shuffle, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, self.H, self.W)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, img, gradients, pred, mask):
        """
        img: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(img)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if self.grad_weighted_loss:
            grad_weights = F.max_pool2d(gradients, self.patch_size).mean(1)
            grad_weights = rearrange(grad_weights, 'b h w -> b (h w)')
            # grad_weights = (grad_weights - torch.amin(grad_weights, keepdim=True)) / \
            #                (torch.amax(grad_weights, keepdim=True) - torch.amin(grad_weights, keepdim=True))
            grad_weights = grad_weights / grad_weights.sum(dim=1, keepdims=True)
            loss = (loss * grad_weights).sum()
        else:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, img, grad_mask=None, mask_ratio=0.75, ids_shuffle=None, ids_restore=None):
        # encoder
        latent, mask, ids_shuffle, ids_restore = self.forward_encoder(img, mask_ratio, ids_shuffle, ids_restore)
        # decoder
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(img, grad_mask, pred, mask)
        if self.training:
            return loss, pred, mask, ids_shuffle, ids_restore
        else:
            return loss, pred, mask, ids_shuffle, ids_restore, self.abnormal_score(img, pred, mask)

    def abnormal_score(self, img, pred, mask):
        img = self.patchify(img)
        if self.use_only_masked_tokens_ab:
            mask = mask.bool()
            selected_pred = []
            selected_lbl = []
            for i in range(0, img.shape[0]):
                selected_pred.append(pred[i][mask[i]])
                selected_lbl.append(img[i][mask[i]])

            pred = torch.stack(selected_pred)
            img = torch.stack(selected_lbl)
        return ((img - pred) ** 2).mean((1, 2))  # MSE

