import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.manual_seed(seed + i)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True