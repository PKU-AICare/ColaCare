import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision
from torchmetrics.classification import BinaryF1Score
from sklearn import metrics as sklearn_metrics
import numpy as np


def minpse(preds, labels):
    precisions, recalls, _ = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score


def get_binary_metrics(preds, labels):
    threshold = 0.5
    
    accuracy = Accuracy(task="binary", threshold=threshold)
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")
    f1 = BinaryF1Score(threshold=threshold)

    # convert labels type to int
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
        preds = torch.tensor(preds)
    labels = labels.type(torch.int)
    accuracy(preds, labels)
    auroc(preds, labels)
    auprc(preds, labels)
    f1(preds, labels)

    # return a dictionary
    return {
        "accuracy": accuracy.compute().item(),
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "minpse": minpse(preds, labels),
        "f1": f1.compute().item(),
    }