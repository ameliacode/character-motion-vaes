import time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

from models import *

class HumorModel(nn.Module):
    def __init__(self,
                 latent_size=48,
                 steps_in = 1
                 ):
        self.steps_in = steps_in
        self.steps_out = 1



