import argparse
from functools import partial

import torch
from torch import nn

from configs.configs import get_configs_avenue, get_configs_shanghai
from data.train_dataset import VadTrainDataset
from model.mae_cvt import MaskedAutoencoderCvT


def mae_cvt_patch16(**kwargs):
    model = MaskedAutoencoderCvT(
        patch_size=16, embed_dim=256, depth=3, num_heads=4,
        decoder_embed_dim=128, decoder_depth=3, decoder_num_heads=4,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_cvt_patch8(**kwargs):
    model = MaskedAutoencoderCvT(
        patch_size=8, embed_dim=256, depth=3, num_heads=4,
        decoder_embed_dim=128, decoder_depth=3, decoder_num_heads=4,
        mlp_ratio=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avenue')
    args = parser.parse_args()
    if args.dataset == 'avenue':
        args = get_configs_avenue()
    else:
        args = get_configs_shanghai()
    vtd = VadTrainDataset(args)

    model = mae_cvt_patch16(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                            use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                            abnormal_score_func=args.abnormal_score_func,
                            grad_weighted_loss=args.grad_weighted_rec_loss).float()
    device = args.device
    model.to(device)
    model.train(True)

    print(f"len: {vtd.__len__()}")  # avenue: 15328
    sampler_train = torch.utils.data.RandomSampler(vtd)
    data_loader_train = torch.utils.data.DataLoader(
        vtd, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    for data_iter_step, (sample1, sample2, grad_mask) in enumerate(data_loader_train):
        if data_iter_step == 1:
            break
        sample1 = sample1.to(device, non_blocking=True)
        sample2 = sample2.to(device, non_blocking=True)
        grad_mask = grad_mask.to(device, non_blocking=True)
        loss, _, _, ids_shuffle, ids_restore = model(sample1, grad_mask=grad_mask,
                                                     mask_ratio=args.mask_ratio)
        loss2, _, _, _, _ = model(sample2, grad_mask=grad_mask, mask_ratio=args.mask_ratio,
                                  ids_shuffle=ids_shuffle, ids_restore=ids_restore)
        print("loss: ", loss.item())
        print("loss: ", loss2.item())
