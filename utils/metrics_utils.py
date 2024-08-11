import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision
from torchmetrics.classification import BinaryF1Score
import numpy as np
from sklearn import metrics as sklearn_metrics


def minpse(preds, labels):
    precisions, recalls, _ = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score


def get_binary_metrics(preds, labels):
    accuracy = Accuracy(task="binary", threshold=0.5)
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")
    f1 = BinaryF1Score()

    # convert labels type to int
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


def check_metric_is_better(cur_best, score, main_metric='auroc'):
    if cur_best == {}:
        return True
    if score > cur_best[main_metric]:
        return True
    return False


def bootstrap(preds, labels, K=100, seed=42):
    """Bootstrap resampling for binary classification metrics. Resample K times"""
    
    length = len(preds)
    # length = N
    np.random.seed(seed)
    
    # Initialize a list to store bootstrap samples
    bootstrapped_samples = []

    # Create K bootstrap samples
    for _ in range(K):
        # Sample with replacement from the indices
        sample_indices = np.random.choice(length, length, replace=True)

        # Use the sampled indices to get the bootstrap sample of preds and labels
        sample_preds = preds[sample_indices]
        sample_labels = labels[sample_indices]
        
        # Store the bootstrap samples
        bootstrapped_samples.append((sample_preds, sample_labels))

    return bootstrapped_samples


def export_metrics(bootstrapped_samples):
    metrics = {"accuracy": [], "auprc": [], "auroc": [], "minpse": [], "f1": []}
    for sample in bootstrapped_samples:
        sample_preds, sample_labels = sample[0], sample[1]
        res = get_binary_metrics(sample_preds, sample_labels)

        for k, v in res.items():
            metrics[k].append(v)

    # convert to numpy array
    for k, v in metrics.items():
        metrics[k] = np.array(v)
    
    # calculate mean and std
    for k, v in metrics.items():
        metrics[k] = {"mean": np.mean(v), "std": np.std(v)}
    return metrics


def run_bootstrap(preds, labels, K=100, seed=42):
    bootstrap_samples = bootstrap(preds, labels, K=K, seed=seed)
    metrics = export_metrics(bootstrap_samples)
    return metrics