import random
import torch
import numpy as np


def setup_seed(seed):
    # random
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # numpy
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



