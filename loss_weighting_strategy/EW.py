import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from loss_weighting_strategy.abstract_weighting import AbsWeighting

default_args_dict = {}


class EW(AbsWeighting):
    r"""Equally Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` means the number of tasks.

    """
    def __init__(self):
        super(EW, self).__init__()
        
    def backward(self, losses, **kwargs):
        loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        loss.backward()
        return loss.item(), np.ones(self.task_num)
