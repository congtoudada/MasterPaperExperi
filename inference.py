import os
from collections.abc import Iterable

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics

from util import misc
from util.abnormal_utils import filt


def inference(model: torch.nn.Module, data_loader: Iterable,
              device: torch.device,
              log_writer=None, args=None):
    model.is_inference = True
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing '

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    predictions = []

    labels = []
    videos = []
    frames = []
    for data_iter_step, (samples, grads, label, vid, frame_name) in enumerate(
            metric_logger.log_every(data_loader, args.print_freq, header)):
        videos += list(vid)
        labels += list(label.detach().cpu().numpy())
        frames += list(frame_name)
        samples = samples.to(device)
        grads = grads.to(device)
        _, _, _, _, _, recon_error_tc = model(samples, grad_mask=grads, mask_ratio=args.mask_ratio)
        recon_error = recon_error_tc.detach().cpu().numpy()
        predictions += list(recon_error)

    # Compute statistics
    predictions = np.array(predictions)
    labels = np.array(labels)
    videos = np.array(videos)

    if args.dataset == 'avenue':
        evaluate_model(predictions, labels, videos,
                       normalize_scores=False,
                       range=38, mu=11)
    else:
        # evaluate_model(predictions_teacher, labels, videos,
        #                normalize_scores=True,
        #                range=900, mu=282)
        evaluate_model(predictions, labels, videos,
                       normalize_scores=True,
                       range=900, mu=282)


def evaluate_model(predictions, labels, videos,
                   range=302, mu=21, normalize_scores=False, save_vis=True):
    aucs = []
    filtered_preds = []
    filtered_labels = []
    for vid in np.unique(videos):
        pred = predictions[np.array(videos) == vid]
        pred = filt(pred, range=range, mu=mu)
        if normalize_scores:
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))

        pred = np.nan_to_num(pred, nan=0.)

        # pred
        # plt.plot(pred)
        # plt.xlabel("Frames")
        # plt.ylabel("Anomaly Score")
        # os.makedirs(f"tmp/graphs", exist_ok=True)
        # plt.savefig(f"tmp/graphs/{vid}.png")
        # plt.close()

        filtered_preds.append(pred)
        lbl = labels[np.array(videos) == vid]
        filtered_labels.append(lbl)

        lbl = np.array([0] + list(lbl) + [1])
        pred = np.array([0] + list(pred) + [1])

        fpr, tpr, thresholds = metrics.roc_curve(lbl, pred)

        res = metrics.auc(fpr, tpr)
        aucs.append(res)

        if save_vis:
            if not normalize_scores:
                pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            # pred + label
            # Plot the anomaly score (pred) and the label (lbl) on the same graph
            plt.plot(pred, label='Anomaly Score', color='b')
            plt.plot(lbl, label='Ground Truth', color='r', linestyle='--')
            plt.xlabel("Frames")
            plt.ylabel("Anomaly Score")
            plt.legend()
            plt.title(f"Anomaly Detection: Video {vid}")

            # Save the plot for this video
            os.makedirs(f"tmp/graphs_labels", exist_ok=True)
            plt.savefig(f"tmp/graphs_labels/{vid}.png")
            plt.close()

    macro_auc = np.nanmean(aucs)

    # Micro-AUC
    filtered_preds = np.concatenate(filtered_preds)
    filtered_labels = np.concatenate(filtered_labels)

    fpr, tpr, _ = metrics.roc_curve(filtered_labels, filtered_preds)
    micro_auc = metrics.auc(fpr, tpr)
    micro_auc = np.nan_to_num(micro_auc, nan=1.0)

    print(f"MicroAUC: {micro_auc}, MacroAUC: {macro_auc}, range:{range}, mu:{mu}, normalize scores:{normalize_scores}")
    # gather the stats from all processes
    return micro_auc, macro_auc
