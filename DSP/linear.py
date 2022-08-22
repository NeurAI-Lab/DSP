import torch
import sys
sys.path.insert(0, '.')
from models.simclr import SimCLR
from util.test import test_all_datasets, initialize, testloaderSimCLR
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(10)
torch.manual_seed(10)


if __name__ == '__main__':
    args, writer = initialize()
    simclr = SimCLR(args)
    state_dict = torch.load(args.model_path, map_location=args.device)
    simclr.load_state_dict(state_dict)
    simclr = simclr.cuda()
    test_all_datasets(args, writer, simclr)
