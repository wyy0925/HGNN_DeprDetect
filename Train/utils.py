from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import random
import torch
import numpy as np
import os

def entropy_minimization_loss(R,miu, eps=1e-10):
    # R 的尺寸是 (batch_size, n, n_r)
    R_log_R = R * torch.log(R + eps)
    entropy_loss = -R_log_R.sum(dim=(1, 2))  # 对 n 和 n_r 维度求和
    return miu*entropy_loss.mean()

def acc_target(predicted_labels,true_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return accuracy,precision,recall,f1


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def l1_regularizer(weight, lambda_l1):
    if lambda_l1 == 0:
        return 0
    else:
        return lambda_l1 * torch.norm(weight, 1)