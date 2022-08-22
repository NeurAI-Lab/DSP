import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
import sys
from time import ctime
import os
sys.path.insert(0, '.')
from util.utils import logger, summary_writer, log
from config.option import Options
from models.simclr import SimCLR
from util.test import testloaderSimCLR, test_all_datasets
from util.utils import save_checkpoint
from transforms.simclr_transform import SimCLRTransform

np.random.seed(10)
random.seed(10)
torch.manual_seed(10)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def train_supervised(args, loader, model, criterion, optimizer, scheduler):
    """
    Train supervised model
    """
    loss_epoch, accuracy_epoch = 0, 0
    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(args.device)
        y = y.to(args.device)

        _, output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_epoch += loss.item()
        if i % 50 == 0:
            log(f"Batch [{i}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}")
    return loss_epoch, accuracy_epoch


if __name__ == "__main__":
    args = Options().parse()
    log_dir = os.path.join(args.save_dir, "{}_bs_{}".format(args.backbone, args.sup_batchsize),
                                ctime().replace(' ', '_'))
    writer = summary_writer(args, log_dir)
    logger(args)
    args.start_time = datetime.now()
    log("Starting at  {}".format(datetime.now()))
    log("arguments parsed: {}".format(args))
    criterion = nn.CrossEntropyLoss()

    model = SimCLR(args)
    model.cuda(args.device)
    transform = SimCLRTransform(size=args.img_size).sup_transform
    train_loader, val_loader, test_loader = testloaderSimCLR(args, args.sup_dataset, transform, args.sup_batchsize, args.sup_data_dir)
    optimizer = SGD(model.parameters(), lr=args.sup_lr,  momentum=0.9, weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=[180], gamma=0.1)
    for epoch in range(1, args.sup_epochs + 1):
        # Train
        loss_epoch, accuracy_epoch = train_supervised(args, train_loader, model, criterion, optimizer, scheduler)
        log(f"Epoch [{epoch}/{args.sup_epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}")

        # Save checkpoint after every epoch
        path = save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch, filename='checkpoint.pth'.format(epoch))
        if os.path.exists:
            state_dict = torch.load(path, map_location=args.device)
            model.load_state_dict(state_dict)

        # Save the model at specific checkpoints
        if epoch % 10 == 0:
            if args.distribute:
                # Save DDP model's module
                save_checkpoint(state_dict=model.module.state_dict(), args=args, epoch=epoch, filename='checkpoint_model_{}.pth'.format(epoch))
            else:
                save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch, filename='checkpoint_model_{}.pth'.format(epoch))

        writer.add_scalar("CrossEntropyLoss/train", loss_epoch / len(train_loader), epoch)

    # Test the supervised Model
    test_all_datasets(args, writer, model)
