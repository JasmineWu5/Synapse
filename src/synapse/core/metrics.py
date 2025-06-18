import numpy as np
import torch

def weighted_accuracy(score: torch.Tensor, true: torch.Tensor, weight: torch.Tensor):
    """Compute weighted accuracy."""
    y_pred = torch.argmax(score, dim=1)
    correct = (y_pred == true).float()
    weighted_correct = correct * weight
    return torch.sum(weighted_correct) / torch.sum(weight)

def weighted_auc_2cls(score: torch.Tensor, true: torch.Tensor,  weight: torch.Tensor):
    """Compute weighted ROC curve accounting for negative weights."""
    score = score.numpy(force=True)
    true = true.numpy(force=True)
    weight = weight.numpy(force=True)
    # Sort by score (descending)
    sorted_indices = np.argsort(-score)
    true = true[sorted_indices]
    weight = weight[sorted_indices]
    # Define signal and background masks
    is_signal = true == 1
    is_background = true == 0
    # Compute total positive and negative weights
    total_signal_weight = np.sum(weight[is_signal])
    total_background_weight = np.sum(weight[is_background])
    # Compute cumulative sums
    tpr = np.cumsum(weight * is_signal) / total_signal_weight
    fpr = np.cumsum(weight * is_background) / total_background_weight
    fpr, tpr = np.concatenate(([0], fpr)), np.concatenate(([0], tpr))  # Ensure starting point (0,0)
    auc = np.trapezoid(tpr, fpr)
    return auc

def weighted_confusion_matrix_2cls(score: torch.Tensor, true: torch.Tensor, weight: torch.Tensor):
    """Compute weighted confusion matrix accounting for negative weights."""
    score = score.numpy(force=True)
    true = true.numpy(force=True)
    weight = weight.numpy(force=True)
    y_pred = score.argmax(1)
    cm = np.zeros((2, 2), dtype=np.float32)
    for i in range(len(true)):
        if true[i] == 1:
            if y_pred[i] == 1:
                cm[0, 0] += weight[i]  # True Positive
            else:
                cm[0, 1] += weight[i]  # False Negative
        else:
            if y_pred[i] == 0:
                cm[1, 1] += weight[i]  # True Negative
            else:
                cm[1, 0] += weight[i]  # False Positive
    # Normalizes confusion matrix over the true (rows)
    cm[0, :] /= cm[0, :].sum() if cm[0, :].sum() > 0 else 1
    cm[1, :] /= cm[1, :].sum() if cm[1, :].sum() > 0 else 1

    return cm


