import argparse
import datetime
import json
import os
import time
import torch
from pathlib import Path
from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
from configs.configs import get_configs_avenue, get_configs_shanghai
from data.test_dataset import VadTestDataset
from data.train_dataset import VadTrainDataset
from engine_train import train_one_epoch, test_one_epoch
from inference import inference
from model.model_factory import mae_cvt_patch16, mae_cvt_patch8
from util import misc


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    log_writer = SummaryWriter(log_dir=args.output_dir)

    device = args.device
    if args.run_type == 'train':
        dataset_train = VadTrainDataset(args)
        print(dataset_train)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    dataset_test = VadTestDataset(args)
    print(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # define the model
    if args.dataset == 'avenue':
        model_target = mae_cvt_patch16(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                       use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                                       abnormal_score_func=args.abnormal_score_func,
                                       grad_weighted_loss=args.grad_weighted_rec_loss).float()
        if args.run_type == "train":
            model_online = mae_cvt_patch16(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                           use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                                           abnormal_score_func=args.abnormal_score_func,
                                           grad_weighted_loss=args.grad_weighted_rec_loss).float()
        else:
            model_online = model_target

    else:
        model_target = mae_cvt_patch8(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                      use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                                      abnormal_score_func=args.abnormal_score_func,
                                      grad_weighted_loss=args.grad_weighted_rec_loss).float()
        if args.run_type == "train":
            model_online = mae_cvt_patch8(norm_pix_loss=args.norm_pix_loss, img_size=args.input_size,
                                          use_only_masked_tokens_ab=args.use_only_masked_tokens_ab,
                                          abnormal_score_func=args.abnormal_score_func,
                                          grad_weighted_loss=args.grad_weighted_rec_loss).float()
        else:
            model_online = model_target
    model_target.is_target = True
    model_online.to(device)
    model_target.to(device)
    # print("Online Model = %s" % str(model_online))
    print("Target Model = %s" % str(model_target))

    if args.run_type == "train":
        do_training(args, data_loader_test, data_loader_train, device, log_writer, model_online, model_target)
    elif args.run_type == "inference":
        model_data = torch.load(args.output_dir + "/checkpoint-best.pth")['model_target']
        model_target.load_state_dict(model_data, strict=False)
        with torch.no_grad():
            inference(model_target, data_loader_test, device, args=args)


def do_training(args, data_loader_test, data_loader_train, device, log_writer, model_online, model_target):
    print("actual lr: %.2e" % args.lr)
    n_parameters = sum(p.numel() for p in model_online.parameters() if p.requires_grad)
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_online, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # momentum parameter is increased to 1. during training with a cosine
    # schedule
    momentum_schedule = misc.cosine_scheduler(
        args.momentum_target, 1, args.epochs, len(data_loader_train)
    )

    misc.load_model(args=args, model_online=model_online, model_target=model_target, optimizer=optimizer,
                    loss_scaler=loss_scaler, momentum_schedule=momentum_schedule)
    # 初始化model_target，freeze
    if not args.resume:
        model_target.load_state_dict(model_online.state_dict())
    for param in model_target.parameters():
        param.requires_grad = False

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_micro = 0.0

    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model_online, model_target, data_loader_train,
            optimizer, device, epoch, loss_scaler, momentum_schedule,
            log_writer=log_writer,
            args=args
        )
        log_stats_train = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        # 每n个epoch评估一次，超过eval epoch将持续评估
        if epoch % 3 == 0 or epoch > args.eval_epoch:
            # 使用目标模型评估
            test_stats = test_one_epoch(
                model_target, data_loader_test, device, epoch, log_writer=log_writer, args=args
            )
            log_stats_test = {**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch}

            if args.output_dir:
                # 保存最新权重
                misc.save_model(args=args, model_online=model_online, model_target=model_target,
                                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                                momentum_schedule=momentum_schedule, latest=True)
                with open(os.path.join(args.output_dir, "log_test.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats_test) + "\n")
            if test_stats['micro'] > best_micro:
                # 保存best权重
                best_micro = test_stats['micro']
                misc.save_model(args=args, model_online=model_online, model_target=model_target,
                                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                                momentum_schedule=momentum_schedule, best=True)

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log_train.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_train) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avenue')
    args = parser.parse_args()
    if args.dataset == 'avenue':
        args = get_configs_avenue()
    else:
        args = get_configs_shanghai()  #
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
