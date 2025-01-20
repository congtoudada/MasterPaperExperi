import math
import sys
import torch
import util.misc as misc
from typing import Iterable
import numpy as np
from util.abnormal_utils import filt
import sklearn.metrics as metrics


def train_one_epoch(model_online: torch.nn.Module,
                    model_target: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, momentum_schedule,
                    log_writer=None, args=None):
    model_online.train()
    model_online = model_online.float()
    metric_logger = misc.MetricLogger(delimiter="")
    header = 'Epoch: [{}]'.format(epoch)

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (sample1, sample2, grad_mask) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        sample1 = sample1.to(device, non_blocking=True)
        sample2 = sample2.to(device, non_blocking=True)
        grad_mask = grad_mask.to(device, non_blocking=True)

        online_loss, online_pred, online_mask, ids_shuffle, ids_restore = model_online(sample1, grad_mask=grad_mask,
                                                                                       mask_ratio=args.mask_ratio)
        with torch.no_grad():
            model_target.eval()
            target_loss, target_pred, _, _, _ = model_target(sample2, grad_mask=grad_mask,
                                                             ids_shuffle=ids_shuffle, ids_restore=ids_restore)
        rec_cons_loss = misc.cons_loss(online_pred, online_mask, target_pred)
        loss = online_loss + target_loss + args.gamma * rec_cons_loss
        loss_value = loss.item()
        online_loss_value = online_loss.item()
        target_loss_value = target_loss.item()
        rec_cons_loss_value = rec_cons_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # loss_scaler(loss, optimizer, parameters=model_online.parameters(), update_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            ms = momentum_schedule[data_iter_step]  # momentum parameter
            for param_q, param_k in zip(
                    model_online.parameters(), model_target.parameters()
            ):
                param_k.data.mul_(ms).add_((1 - ms) * param_q.detach().data)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value)
        metric_logger.update(online_loss=online_loss_value)
        metric_logger.update(target_loss=target_loss_value)
        metric_logger.update(rec_cons_loss=rec_cons_loss_value)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        online_loss_value_reduce = misc.all_reduce_mean(online_loss_value)
        target_loss_value_reduce = misc.all_reduce_mean(target_loss_value)
        rec_cons_loss_value_reduce = misc.all_reduce_mean(rec_cons_loss_value)
        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.update(train_loss=online_loss_value_reduce, head="online_loss", step=epoch_1000x)
            log_writer.update(train_loss=target_loss_value_reduce, head="target_loss", step=epoch_1000x)
            log_writer.update(train_loss=rec_cons_loss_value_reduce, head="rec_cons_loss", step=epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_one_epoch(model: torch.nn.Module,
                   data_loader: Iterable,
                   device: torch.device, epoch: int,
                   log_writer=None, args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    predictions = []
    labels = []
    videos = []
    for data_iter_step, (sample, grads, label, vid, _) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())

        sample = sample.to(device)
        grads = grads.to(device)
        _, _, _, _, _, recon_error = model(sample, grad_mask=grads, mask_ratio=args.mask_ratio)

        recon_error = recon_error.detach().cpu().numpy()
        predictions += list(recon_error)

    # Compute statistics
    predictions = np.array(predictions)
    labels = np.array(labels)
    videos = np.array(videos)

    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        pred = np.nan_to_num(pred, nan=0.)
        if args.dataset=='avenue':
            pred = filt(pred, range=38, mu=11)
        else:
            raise ValueError('Unknown parameters for predictions postprocessing')
        # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        filtered_preds.append(pred)
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)
        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])
        fpr, tpr, _ = metrics.roc_curve(lbl, pred)
        res = metrics.auc(fpr, tpr)
        aucs.append(res)

    macro_auc = np.nanmean(aucs)

    # Micro-AUC
    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    # gather the stats from all processes
    print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}")
    return {"micro": micro_auc, "macro": macro_auc}
