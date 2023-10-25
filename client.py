from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict
import warnings
import torch.nn as nn
from utils import (
    construct_model,
    construct_loss_func,
    construct_optimizer,
    get_model_params,
    set_seed,
)
from flwr.common import NDArrays
from engine import train, evalulate
from data import construct_dataset
import yaml
from copy import deepcopy

warnings.filterwarnings("ignore")


class LiGOClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        device: str,
        args: dict = None,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.args = args

    def set_parameters(self, parameters: NDArrays, config: dict):
        """Loads a efficientnet model and replaces it parameters with the ones given."""
        model = construct_model(config["model"], config["homogeneous_model_kwargs"])
        if parameters is not None:
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
        return model

    def train_small_model(self, config):
        # define the small model kwargs
        config = deepcopy(config)
        wo_config = deepcopy(config)
        del wo_config["model_kwargs"]["target_hiddens"]
        del wo_config["model_kwargs"]["target_layers"]
        wo_model = construct_model(config["model"], wo_config["model_kwargs"])
        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["small_model_training_round"]

        # Construct Data Loader for training
        trainLoader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )
        config["optimizer_kwargs"]["params"] = wo_model.parameters()
        criterion = construct_loss_func(config["criterion"], config["criterion_kwargs"])
        optimizer = construct_optimizer(config["optimizer"], config["optimizer_kwargs"])
        results = train(
            wo_model, criterion, trainLoader, optimizer, epochs, self.device
        )

        # Begin train the ligo operator
        model = construct_model(config["model"], config["model_kwargs"])
        model.load_state_dict(wo_model.state_dict(), strict=False)
        config["optimizer_kwargs"]["params"] = model.parameters()
        criterion = construct_loss_func(config["criterion"], config["criterion_kwargs"])
        optimizer = construct_optimizer(config["optimizer"], config["optimizer_kwargs"])
        results = train(model, criterion, trainLoader, optimizer, epochs, self.device)

        num_examples_train = len(self.trainset)
        parameters_prime = get_model_params(model, mode=1)
        return parameters_prime, num_examples_train, results

    def fit(self, parameters: NDArrays, config: dict):
        """Train parameters on the locally held training set."""
        config = deepcopy(config)

        if config["small_model_training"]:
            return self.train_small_model(config)

        # Update local model parameters
        model = self.set_parameters(parameters, config)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_ep"]

        # Construct Data Loader for training
        trainLoader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )
        # testLoader = DataLoader(self.testset, batch_size=batch_size, pin_memory=True)
        # Construct loss function and optimizer for training
        config["optimizer_kwargs"]["params"] = model.parameters()
        criterion = construct_loss_func(config["criterion"], config["criterion_kwargs"])
        optimizer = construct_optimizer(config["optimizer"], config["optimizer_kwargs"])
        # Conduct training on the given datset
        results = train(model, criterion, trainLoader, optimizer, epochs, self.device)
        # Extracy the numpy array format of the parameters for the model
        parameters_prime = get_model_params(model)
        # Extract the training sample number for Weighted AVG.
        num_examples_train = len(self.trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters: NDArrays, config: dict):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters, config)

        # Get config values
        batch_size: int = config["batch_size"]
        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(
            self.testset, batch_size=batch_size, pin_memory=True, drop_last=False
        )
        # Perform evaluation on the test set.
        loss, result = evalulate(model, testloader, self.device)
        return float(loss), len(self.testset), result


def client_dry_run(args: dict, device: str = "cpu"):
    """Weak tests to check whether all client methods are working as expected."""
    args = deepcopy(args)
    args["small_model_training"] = False
    args["model_kwargs"] = args["model_kwargs"]["tiny"]
    model = construct_model(args["model"], args["homogeneous_model_kwargs"])
    trainset, testset = construct_dataset(
        args["dataset"], args["data_root"], args["num_clients"], args["iid_degree"], 0
    )
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    client = LiGOClient(trainset, testset, device, args=None)
    client.fit(
        None,
        args,
    )

    client.evaluate(get_model_params(model), args)
    client.train_small_model(args)
    print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
        help="Config_path of the client, should be consistent across all clients.",
    )
    parser.add_argument(
        "--dry",
        type=bool,
        default=False,
        required=False,
        help="Do a dry-run to check the client",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        required=False,
        help="Specifies the artificial data partition of training dataset to be used. \
        Picks partition 0 by default",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        required=False,
        help="Select the device to perform calculation",
    )

    args = parser.parse_args()
    # device
    device = torch.device(args.device)
    # Load config file.
    with open(args.cfg_path, "r") as f:
        fedligo_cfg = yaml.load(f, Loader=yaml.FullLoader)
    set_seed(fedligo_cfg["seed"])
    if args.dry:
        client_dry_run(fedligo_cfg, device)
    else:
        # Load a subset of CIFAR-10 to simulate the local data partition
        trainset, testset = construct_dataset(
            fedligo_cfg["dataset"],
            fedligo_cfg["data_root"],
            fedligo_cfg["num_clients"],
            fedligo_cfg["iid_degree"],
            args.partition,
        )

        # Start Flower client
        client = LiGOClient(trainset, testset, device)

        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
