import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from models.ViT import LiGOViT
from models.MLP import LiGOMLP
import warnings
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from flwr.common import NDArrays

warnings.filterwarnings("ignore")
model_dict = {"LiGOMLP": LiGOMLP, "LiGOViT": LiGOViT}


def construct_model(model_name: str, kwargs: dict) -> nn.Module:
    """construct a model according to the given parameters

    Args:
        model_name (str): the model name
        kwargs (dict): the keword arguments to be passed to the model
    """
    kwargs = {} if kwargs is None else kwargs
    return model_dict[model_name](**kwargs)


def construct_optimizer(optimizer_name: str, kwargs: dict) -> optim.Optimizer:
    """construct optimizer

    Args:
        optimizer_name (str): the name of the optimizer
        kwargs (dict): kwargs of the optimizer like weight_decay(l2_penalty)

    Returns:
        torch.optim.Optimizer: the optimizer
    """
    kwargs = {} if kwargs is None else kwargs
    optimizer = getattr(optim, optimizer_name)(**kwargs)
    return optimizer


def construct_loss_func(func_name: str, kwargs: dict) -> nn.Module:
    """construct the loss function

    Args:
        func_name (str): loss function name
        kwargs (dict): loss funciton kwargs like "reduction=sum"

    Returns:
        nn.Module: the constructed loss_func.
    """
    kwargs = {} if kwargs is None else kwargs
    loss_func = getattr(nn, func_name)(**kwargs)
    return loss_func


def gen_hetro_model_args(args: dict) -> dict:
    """Generates heterogeneous model architectures according to arguments.

    Args:
        args (dict): the arguments.
    """
    args = deepcopy(args)
    n_type_models = len(args["model_kwargs"].keys())
    keys = list(args["model_kwargs"].keys())
    model_type = np.random.choice(np.arange(n_type_models))
    args["model_kwargs"] = args["model_kwargs"][keys[model_type]]
    return args


def get_model_params(model: nn.Module, mode: int = 0) -> NDArrays:
    """Returns a model's parameters.

    Args:
        model (nn.Module): _description_
        mode (int, optional): _description_. Defaults to 0: state_dict(), 1:expanded_model_state_dict, 2:ligo_dict

    Returns:
        NDArrays: list of model's parameters (ndarray format)
    """
    if mode == 0:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
    elif mode == 1:
        return [val.cpu().numpy() for _, val in model.expanded_model_state_dict.items()]
    elif mode == 2:
        return [val.cpu().numpy() for _, val in model.ligo_dict.items()]
    else:
        raise ValueError("Invalid mode")


def set_seed(seed: int):
    """set seed for numpy and pytorch

    Args:
        seed (int): the random seed
    """
    torch.manual_seed(seed)  # set the seed for Pytorch
    np.random.seed(seed)  # set the seed for numpy
    torch.backends.cudnn.benchmark = False  # disable cuDNN benchmarking
