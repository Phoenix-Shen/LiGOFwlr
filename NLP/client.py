from torch.utils.data import DataLoader
import torchvision.datasets
from torch.utils.data import Dataset
import torch
import flwr as fl
import argparse
from collections import OrderedDict
from data.base_data_manager import BaseDataLoader
from model_args import ModelArgs
from flwr.common import NDArrays
from utils import (
    construct_model,
    construct_trainer,
    get_model_params,
    construct_dataset,
    set_seed,
    load_args,
)
from logging import DEBUG, INFO
from flwr.common import log
from typing import Dict
import torch.nn as nn
from copy import deepcopy
import yaml


class LiGOClient(fl.client.NumPyClient):
    def __init__(
        self,
        train_loader: BaseDataLoader,
        test_loader: BaseDataLoader,
        device: str,
        args: ModelArgs,
        idx: int,
    ):
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.idx = idx

    def set_parameters(self, parameters: NDArrays) -> nn.Module:
        """Loads a deep learning model and replaces it parameters with the ones given.

        Args:
            parameters (NDArrays): the distributed parameters

        Returns:
            nn.Module: the constructed model
        """
        model = construct_model(
            self.args.model_type,
            self.args.model_class,
            self.args.homogeneous_model_kwargs,
        )
        if len(parameters) > 0:
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
        self.args.model_kwargs = self.args.model_kwargs[
            list(self.args.model_kwargs.keys())[training_instruction["small_model_idx"]]
        ]

        # Define the small model (without ligo operator)
        wo_model = construct_model(
            self.args.model_type,
            self.args.model_class,
            {"small_model": self.args.model_kwargs["small_model"], "large_model": None},
        )
        # Define the small model trainer
        wo_model_trainer = construct_trainer(
            self.args.model_type,
            self.args.model_class,
        )
        # Train the small model
        wo_args = deepcopy(self.args)
        wo_args.epochs = wo_args.small_model_training_round
        results = wo_model_trainer.train(
            wo_model, self.train_loader, self.device, wo_args
        )

        log(
            INFO,
            "Client {}, small model traning finished, loss is {}.".format(
                self.idx, results
            ),
        )
        # Begin to train the ligo operator
        model = construct_model(
            self.args.model_type, self.args.model_class, self.args.model_kwargs
        )
        model.load_state_dict(wo_model.state_dict(), strict=False)

        results = wo_model_trainer.train(model, self.train_loader, self.device, wo_args)
        eval_result = wo_model_trainer.test(
            model, self.test_loader, self.device, self.args
        )
        log(
            INFO,
            "Client {}, ligo traning finished, loss is {}, eval result is {}".format(
                self.idx, results, eval_result
            ),
        )
        num_examples_train = len(self.train_loader.dataset)

        model_homo = construct_model(
            self.args.model_type,
            self.args.model_class,
            self.args.homogeneous_model_kwargs,
        )
        model_homo.load_state_dict(model.expanded_model_state_dict, strict=False)
        parameters_prime = get_model_params(model_homo)
        # log
        log(
            INFO,
            "Client {} finished training the small model and ligo operator.".format(
                self.idx
            ),
        )
        torch.save(model.expanded_model_state_dict, "client_{}.pth".format(self.idx))
        return parameters_prime, num_examples_train, {"traning_loss": results}

    def fit(self, parameters: NDArrays, training_instruction: dict):
        """Train parameters on the locally held training set."""
        log(
            INFO,
            "Client {} received fit signal: {}".format(self.idx, training_instruction),
        )
        # Override the small model training argument
        self.args.small_model_training = training_instruction["small_model_training"]

        if self.args.small_model_training:
            return self.train_small_model(training_instruction)

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Conduct training on the given datset
        trainer = construct_trainer(
            self.args.model_type,
            self.args.model_class,
        )
        results = trainer.train(model, self.train_loader, self.device, self.args)

        # Extracy the numpy array format of the parameters for the model
        parameters_prime = get_model_params(model)
        # Extract the training sample number for Weighted AVG.
        num_examples_train = len(self.train_loader.dataset)
        log(
            INFO,
            "Client {} 's fit process finished, metrics:{}".format(
                self.idx, {"traning_loss": results}
            ),
        )
        return parameters_prime, num_examples_train, {"traning_loss": results}

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)
        trainer = construct_trainer(
            self.args.model_type,
            self.args.model_class,
        )
        eval_result: dict = trainer.test(
            model, self.test_loader, self.device, self.args
        )
        test_loss = eval_result.pop("test_loss")
        return test_loss, len(self.test_loader.dataset), eval_result


def client_dry_run(args: ModelArgs, device: str = "cpu", idx: int = 0):
    """Weak tests to check whether all client methods are working as expected.
    Args:
        args (ModelArgs): the arguments need to be passed into
        device (str, optional): Defaults to "cpu".
        idx (int, optional): Defaults to 0, the idx of the client.
    """
    model = construct_model(
        args.model_type, args.model_class, args.homogeneous_model_kwargs
    )

    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = construct_dataset(args)
    print(class_num)
    train_loader, test_loader = train_data_local_dict[idx], test_data_local_dict[idx]
    client = LiGOClient(train_loader, test_loader, device, args, idx)

    parameters, num_examples, train_results = client.fit(
        None,
        {"small_model_idx": 2, "small_model_training": True},
    )
    print(num_examples, train_results)

    parameters, num_examples, train_results = client.fit(
        parameters,
        {"small_model_idx": 2, "small_model_training": False},
    )

    print(client.evaluate(parameters, {}))

    print("Dry Run Successful")


def main() -> None:
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
        default="cuda:0",
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
    # Device
    device = torch.device(args.device)
    # Load config from file
    with open(args.cfg_path, "r") as f:
        modelargs_dict = yaml.load(f, Loader=yaml.FullLoader)
    modelargs = load_args(modelargs_dict)
    modelargs.update_from_dict(modelargs_dict)
    # Set random seed
    set_seed(modelargs.manual_seed)
    # Run the client according to the settings

    if args.dry:
        client_dry_run(modelargs, device, args.partition)
    else:
        # Load data
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = construct_dataset(args)
        train_loader, test_loader = (
            train_data_local_dict[args.partition],
            test_data_local_dict[args.partition],
        )
        # Start flower client.
        client = LiGOClient(
            train_loader, test_loader, device, modelargs, args.partition
        )

        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
