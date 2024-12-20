import torch
from models.ViT import LiGOViT
import warnings
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from flwr.common import NDArrays, EvaluateRes, FitRes
from typing import Dict, Union
import wandb

Scalar = Union[bool, bytes, float, int, str]
# warnings.filterwarnings("ignore")
model_dict = {"LiGOViT": LiGOViT}


def construct_model(model_name: str, kwargs: dict) -> nn.Module:
    """construct a model according to the given parameters

    Args:
        model_name (str): the model name
        kwargs (dict): the keword arguments to be passed to the model
    """
    kwargs = {} if kwargs is None else kwargs
    return model_dict[model_name](**kwargs)


def construct_optimizer(
    model: nn.Module, optimizer_name: str, kwargs: dict
) -> optim.Optimizer:
    """construct optimizer

    Args:
        optimizer_name (str): the name of the optimizer
        kwargs (dict): kwargs of the optimizer like weight_decay(l2_penalty)

    Returns:
        torch.optim.Optimizer: the optimizer
    """
    kwargs = deepcopy(kwargs)
    lr = kwargs.pop("lr")
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "depth_expansion_operator" not in n
            ],
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "depth_expansion_operator" in n
            ],
            "lr": lr,
        },  # 10 times the original lr
    ]
    kwargs = {} if kwargs is None else kwargs
    optimizer = getattr(optim, optimizer_name)(param_groups, **kwargs)
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

    return int(model_type)


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
    torch.cuda.manual_seed_all(seed)  # Set cuda manual seed


def weighted_metrics_avg(metrics: list[tuple[Dict[str, Scalar], int]]) -> dict:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for _, num_examples in metrics])
    # Initialize an empty dictionary to store the aggregated metrics
    aggregated_metrics = {}
    # Loop over the keys of the metrics dictionary
    if len(metrics) > 0:
        for key in metrics[0][0].keys():
            # Calculate the weighted average of the metric values from all clients
            weighted_sum = sum(
                [metric[key] * num_examples for metric, num_examples in metrics]
            )
            weighted_avg = weighted_sum / num_total_evaluation_examples
            # Store the weighted average value in the aggregated metrics dictionary
            aggregated_metrics[key] = weighted_avg
    # Return the aggregated metrics dictionary
    return aggregated_metrics


def log_wandb(results: Union[list[FitRes], list[EvaluateRes]]) -> None:
    """Log to wandb

    Args:
        results (Union[FitRes,EvaluateRes]): the result collected by the server
    """
    # for different types of Result, we have different manners to handle that.
    if isinstance(results[0], FitRes):
        tmp_dict = {}
        for idx, result in enumerate(results):
            for key in result.metrics.keys():
                tmp_dict["client#{}".format(idx) + key] = result.metrics[key]
        wandb.log(tmp_dict)

    if isinstance(results[0], EvaluateRes):
        tmp_dict = {}
        for idx, result in enumerate(results):
            for key in result.metrics.keys():
                tmp_dict["client#{}".format(idx) + key] = result.metrics[key]
            tmp_dict["client#{}".format(idx) + "eval_loss"] = result.loss
        wandb.log(tmp_dict)
