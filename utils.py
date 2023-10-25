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


def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    trainset, testset, num_examples = load_data()
    n_train = int(num_examples["trainset"] / 10)
    n_test = int(num_examples["testset"] / 10)

    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)


def train(net, trainloader, valloader, epochs, device: str = "cpu"):
    """Train the network on the training set."""
    print("Starting training...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader, steps: int = None, device: str = "cpu"):
    """Validate the network on the entire test set."""
    print("Starting evalutation...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
    accuracy = correct / len(testloader.dataset)
    net.to("cpu")  # move model back to CPU
    return loss, accuracy


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
