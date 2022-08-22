import torch
print(torch.__version__)
import numpy as np
import random
from datetime import datetime
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.insert(0, '.')
from util.utils import logger, summary_writer, log
from util.train_util import trainSSL, get_criteria
from config.option import Options
from models.simclr import SimCLR
from optimizers.lars import LARC

np.random.seed(10)
random.seed(10)
torch.manual_seed(10)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    args = Options().parse()
    args.writer = summary_writer(args)
    logger(args)
    args.start_time = datetime.now()
    log("Starting at  {}".format(datetime.now()))
    log("arguments parsed: {}".format(args))
    criterion = get_criteria(args)
    if args.ssl_model == 'simclr':
        model = SimCLR(args)
        if args.optimizer == 'lars':
            optimizer_= SGD(model.parameters(), lr=args.ssl_lr)
            optimizer = LARC(optimizer_)
            if args.scheduler:
                scheduler = CosineAnnealingLR(optimizer_, T_max=100, eta_min=3e-4)
                trainSSL(args, model, optimizer, criterion,  args.writer, scheduler)
        else:
            optimizer = Adam(model.parameters(), lr=args.ssl_lr, weight_decay=1e-6)
            trainSSL(args, model, optimizer, criterion,  args.writer)

