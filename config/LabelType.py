from enum import Enum
import torch.nn as nn


class LabelType(Enum):
    Binary_Classification = 0
    Multiple_Classification = 1
    Regression = 2

