import torch
import torch.nn as nn
from typing import Iterable
from torch import optim
from timm.utils import accuracy
import numpy as np
import sys

# Deprecated due to the backward process is different from the conventional deep learning.
# from timm.utils import NativeScaler

"""
Train and eval functions.
"""


def train(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
):
    """Train the model on the given data_loader for E epoch.

    Args:
        model (nn.Module): _description_
        criterion (nn.Module): _description_
        data_loader (Iterable): _description_
        optimizer (optim.Optimizer): _description_
        epoch (int): _description_
        device (torch.device): _description_
    Returns:
        the AVG training loss
    """
    # Switch to training mode
    model.to(device)
    model.train()
    losses = []
    # Start the main loop
    for ep in range(epoch):
        # Start the inner loop (batch)
        for samples, targets in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # use auto mixed precision to accerate the training procedure.
            with torch.cuda.amp.autocast():
                outputs = model.forward(samples)
                loss = criterion.forward(outputs, targets)
            # If the loss value is infinite, stop
            loss_value = loss.item()
            if not np.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            # optimizer cleans up and step
            optimizer.zero_grad()
            torch.cuda.synchronize()
            loss.backward()

            # For LiGOViT
            if hasattr(model, "backward_param"):
                model.backward_param()

            optimizer.step()

            # Record the loss
            losses.append(loss_value)
    return np.mean(losses)


@torch.no_grad()
def evalulate(model: nn.Module, data_loader: Iterable, device: torch.device):
    """Evaluate the model on the given data_loader.

    Args:
        model (nn.Module): _description_
        data_loader (Iterable): _description_
        device (torch.device): _description_
    Returns:
        the test loss and a dictionary containing the training acc and other metrics.
    """
    # switch to eval model
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses = []
    results: dict = {
        "acc_top5": [],
        "acc_top1": [],
    }
    # start the main loop
    for images, target in data_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # comput the outputs
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        # calculate the top1 and top5 accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_value = loss.item()
        # log
        losses.append(loss_value)
        results["acc_top1"].append(acc1.item())
        results["acc_top5"].append(acc5.item())

    loss_final = np.mean(losses)
    results["acc_top1"] = np.mean(results["acc_top1"])
    results["acc_top5"] = np.mean(results["acc_top5"])
    return loss_final, results
