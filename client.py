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
from logging import DEBUG, INFO
from flwr.common import log
from typing import Dict


warnings.filterwarnings("ignore")


class LiGOClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        device: str,
        args: dict = None,
        idx: int = 0,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.config = args
        self.idx = idx
        self.num_workers = 4

        self.trainLoader = DataLoader(
            self.trainset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            # persistent_workers=True,
        )
        self.testloader = DataLoader(
            self.testset,
            batch_size=self.config["batch_size"],
            pin_memory=True,
            drop_last=False,
            num_workers=self.num_workers,
            # persistent_workers=True,
        )

    def set_parameters(self, parameters: NDArrays) -> nn.Module:
        """Loads a deep learning model and replaces it parameters with the ones given.

        Args:
            parameters (NDArrays): the distributed parameters

        Returns:
            nn.Module: the constructed model
        """
        config = self.config
        model = construct_model(config["model"], config["homogeneous_model_kwargs"])
        if parameters is not None and len(parameters) != 0:
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
        return model

    def train_small_model(
        self, training_instruction: dict
    ) -> tuple[NDArrays, int, Dict[str, float]]:
        """train the small model and it's specific ligo operator

        Args:
            training_instruction (dict): the training configuration from the server.

        Returns:
            tuple[NDArrays,int,Dict[str,float]]: the training results, containing the parameters,
            the number of samples (for aggregation), the results dictionary.
        """
        log(
            INFO,
            "Client {} begin training the small model and ligo operator.".format(
                self.idx
            ),
        )
        # Override the configs, IMPORTANT OPEATION
        self.config["model_kwargs"] = self.config["model_kwargs"][
            list(self.config["model_kwargs"].keys())[
                training_instruction["small_model_idx"]
            ]
        ]
        config = deepcopy(self.config)
        # define the small model kwargs
        wo_config = deepcopy(config)
        del wo_config["model_kwargs"]["target_hiddens"]
        del wo_config["model_kwargs"]["target_layers"]
        wo_model = construct_model(config["model"], wo_config["model_kwargs"])
        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["small_model_training_round"]

        # Construct Data Loader for training
        # trainLoader = DataLoader(
        #     self.trainset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     pin_memory=True,
        #     drop_last=False,
        #     num_workers=self.num_workers,
        # )
        config["optimizer_kwargs"]["params"] = wo_model.parameters()
        criterion = construct_loss_func(config["criterion"], config["criterion_kwargs"])
        optimizer = construct_optimizer(config["optimizer"], config["optimizer_kwargs"])
        results = train(
            wo_model, criterion, self.trainLoader, optimizer, epochs, self.device
        )
        eval_loss, eval_mat = evalulate(wo_model, self.testloader, self.device)
        # Begin to train the ligo operator
        model = construct_model(config["model"], config["model_kwargs"])
        model.load_state_dict(wo_model.state_dict(), strict=False)
        config["optimizer_kwargs"]["params"] = model.parameters()
        criterion = construct_loss_func(config["criterion"], config["criterion_kwargs"])
        optimizer = construct_optimizer(config["optimizer"], config["optimizer_kwargs"])
        results = train(
            model, criterion, self.trainLoader, optimizer, epochs, self.device
        )
        eval_loss, eval_mat = evalulate(model, self.testloader, self.device)
        log(
            INFO,
            "Client {} finished training the small model. Small model test loss:{}, acc:{}".format(
                self.idx,
                eval_loss,
                eval_mat,
            ),
        )
        num_examples_train = len(self.trainset)

        model_homo = construct_model(
            config["model"], config["homogeneous_model_kwargs"]
        )
        model_homo.load_state_dict(model.expanded_model_state_dict, strict=False)
        parameters_prime = get_model_params(model_homo)
        # log
        log(
            INFO,
            "Client {} finished training the small model and ligo operator. Intermidiate model test loss:{}, acc:{}".format(
                self.idx,
                eval_loss,
                eval_mat,
            ),
        )
        return parameters_prime, num_examples_train, {"traning_loss": results}

    def fit(self, parameters: NDArrays, training_instruction: dict):
        """Train parameters on the locally held training set."""
        log(
            INFO,
            "Client {} received fit signal: {}".format(self.idx, training_instruction),
        )
        # Override the small model training argument
        self.config["small_model_training"] = training_instruction[
            "small_model_training"
        ]

        if self.config["small_model_training"]:
            return self.train_small_model(training_instruction)

        config = deepcopy(self.config)
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_ep"]

        # Construct Data Loader for training
        # trainLoader = DataLoader(
        #     self.trainset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     pin_memory=True,
        #     drop_last=False,
        #     num_workers=self.num_workers,
        # )

        # Construct loss function and optimizer for training
        config["optimizer_kwargs"]["params"] = model.parameters()
        criterion = construct_loss_func(config["criterion"], config["criterion_kwargs"])
        optimizer = construct_optimizer(config["optimizer"], config["optimizer_kwargs"])
        # Conduct training on the given datset
        results = train(
            model, criterion, self.trainLoader, optimizer, epochs, self.device
        )
        # Extracy the numpy array format of the parameters for the model
        parameters_prime = get_model_params(model)
        # Extract the training sample number for Weighted AVG.
        num_examples_train = len(self.trainset)
        log(
            INFO,
            "Client {} 's fit process finished, metrics:{}".format(
                self.idx, {"traning_loss": results}
            ),
        )
        return parameters_prime, num_examples_train, {"traning_loss": results}

    def evaluate(self, parameters: NDArrays, config: dict):
        """Evaluate parameters on the locally held test set."""
        log(INFO, "Client {} received evaluation signal:{}".format(self.idx, config))
        config = deepcopy(self.config)
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config values
        batch_size: int = config["batch_size"]
        # Evaluate global model parameters on the local test data and return results
        # testloader = DataLoader(
        #     self.testset,
        #     batch_size=batch_size,
        #     pin_memory=True,
        #     drop_last=False,
        #     num_workers=self.num_workers,
        # )
        # Perform evaluation on the test set.
        loss, result = evalulate(model, self.testloader, self.device)
        log(
            INFO,
            "Client {} 's evaluation finished, loss:{}, metrics:{}.".format(
                self.idx, float(loss), result
            ),
        )
        return float(loss), len(self.testset), result


def client_dry_run(args: dict, device: str = "cuda:0"):
    """Weak tests to check whether all client methods are working as expected."""
    args = deepcopy(args)
    model = construct_model(args["model"], args["homogeneous_model_kwargs"])
    trainset, testset = construct_dataset(
        args["dataset"], args["data_root"], args["num_clients"], args["iid_degree"], 0
    )

    client = LiGOClient(trainset, testset, device, args=args, idx=0)

    param, _, _ = client.fit(
        None,
        {"small_model_idx": 0, "small_model_training": True},
    )
    client.fit(
        param,
        {"small_model_idx": 0, "small_model_training": False},
    )

    client.evaluate(param, {})

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
        default="cuda",
        required=False,
        help="Select the device to perform calculation",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8080",
        help="port",
    )
    args = parser.parse_args()
    # device
    device = torch.device(args.device)
    # Load config file.
    with open(args.cfg_path, "r") as f:
        fedligo_cfg = yaml.load(f, Loader=yaml.FullLoader)
    set_seed(fedligo_cfg["seed"])

    # Run the client according to the settings
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

        # trainset = torch.utils.data.Subset(trainset, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        # testset = torch.utils.data.Subset(testset, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        # Start Flower client
        client = LiGOClient(
            trainset,
            testset,
            device,
            fedligo_cfg,
            args.partition,
        )

        fl.client.start_numpy_client(
            server_address=f"127.0.0.1:{args.port}", client=client
        )


if __name__ == "__main__":
    main()
