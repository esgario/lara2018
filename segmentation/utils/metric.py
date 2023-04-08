# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import torch
import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].type(torch.int) + label_pred[mask].type(torch.int),
        minlength=n_class**2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return (
        {
            "overall acc": acc,
            "mean acc": acc_cls,
            "freqw acc": fwavacc,
            "mean iou": mean_iu,
        },
        cls_iu,
    )


def eval_metric(label_trues, label_preds, n_class):
    label_preds = torch.max(label_preds, 1)[1]

    if len(label_trues.shape) > 2:
        acc, miou = [], []
        for i in range(label_trues.shape[0]):
            score = scores(label_trues[i].cpu(), label_preds[i].cpu(), n_class)[0]
            miou.append(score["mean iou"])
            acc.append(score["overall acc"])
        return np.mean(miou), np.mean(acc)
    else:
        score = scores(label_trues.cpu(), label_preds.cpu(), n_class)[0]
        return score["mean iou"], score["overall acc"]
