import os
import yaml
from easydict import EasyDict
import random
import cv2
import numpy as np
import torch
import time
import logging
from pathlib import Path
from torch.nn import functional as F
import torch
import shutil
from collections import OrderedDict
from sklearn import metrics

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))

def DiceAccuracy(score, target, n_classes=4, sigmoid=True, softmax=False, thresh=0.5):
    if softmax:
        score = torch.softmax(score, dim=1).detach()
    elif sigmoid:
        score = torch.sigmoid(score).detach()
    smooth = 1e-5
    score[score>=thresh] = 1
    score[score<thresh] = 0
    if n_classes!=1:
        target_list = []
        for i in range(n_classes):
            temp_prob = target == i  # * torch.ones_like(input_tensor)
            target_list.append(temp_prob)
        target = torch.stack(target_list, dim=1)
    score = flatten(score)
    target = flatten(target)
    intersect = torch.sum(score * target.float(), dim=-1)
    y_sum = torch.sum(target, dim=-1)
    z_sum = torch.sum(score, dim=-1)
    tmp = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return tmp.mean()

def IoUAccuracy(score, target, eps=1e-6):
    _, D, _ = score.shape
    iou_acc = 0
    for idx in range(D):
        box1_area = torch.abs((score[:, idx, 2]-score[:, idx, 0]) * (score[:, idx, 3] - score[:, idx, 1]))
        box2_area = torch.abs((target[:, idx, 2]-target[:, idx, 0]) * (target[:, idx, 3] - target[:, idx, 1]))
        
        inter_min_x = torch.max(score[:, idx, 0],target[:, idx, 0])
        inter_min_y = torch.max(score[:, idx, 1],target[:, idx, 1])
        inter_max_x = torch.min(score[:, idx, 2],target[:, idx, 2])
        inter_max_y = torch.min(score[:, idx, 3],target[:, idx, 3])    
        
        inter = torch.clamp((inter_max_x - inter_min_x), min=0) * torch.clamp((inter_max_y - inter_min_y), min=0)
        union = box1_area + box2_area - inter
        iou_acc += (inter / (union+eps)).mean()
    return iou_acc / D

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)