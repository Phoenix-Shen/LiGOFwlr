import torch
from client import LiGOClient
import flwr as fl
from strategy import FedLiGO
import argparse
import yaml
from utils import set_seed, load_args, construct_dataset
import wandb
import os
import time
from flwr.common import log
from logging import INFO


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
        help="Config_path of the client, should be consistent across all clients.",
    )
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0")  # Try "cuda" to train on GPU
    print(
        f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 0.2, "num_cpus": 3}

    # Load configuration
    with open(args.cfg_path, "r") as f:
        modelargs_dict = yaml.load(f, Loader=yaml.FullLoader)
        modelargs = load_args(modelargs_dict)
        modelargs.update_from_dict(modelargs_dict)

    set_seed(modelargs.manual_seed)

    # config log file
    log_path = os.path.join(modelargs.output_dir, modelargs.dataset)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(
        log_path,
        time.strftime(
            "%Y-%m-%d_%H_%M_%S",
        )
        + ".log",
    )
    fl.common.logger.configure(identifier=modelargs.dataset, filename=log_file)

    # Load dataset
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = construct_dataset(modelargs)
    log(
        INFO,
        "successfully loaded dataset {}, the class number is {}, the client number is {}".format(
            modelargs.dataset, class_num, len(train_data_local_dict)
        ),
    )

    def client_fn(cid) -> LiGOClient:
        idx = int(cid)
        # Start Flower client
        return LiGOClient(
            train_data_local_dict[idx],
            test_data_local_dict[idx],
            DEVICE,
            modelargs,
            idx=int(cid),
        )

    # init wandb
    local_ep = modelargs.epochs
    batch_size = modelargs.train_batch_size
    iid_degree = modelargs.partition_method
    aggregation = "AGG" if modelargs.aggregation else "NOAGG"

    wandb.init(
        project="FedLiGO",
        name=f"{aggregation},iid_deg{iid_degree},local_ep{local_ep},batch_size{batch_size}",
        # Track hyperparameters and run metadata
        config=modelargs_dict,
    )

    # start the simulation process
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=modelargs.client_num_in_total,
        config=fl.server.ServerConfig(num_rounds=modelargs.comm_round),
        strategy=FedLiGO(modelargs),  # <-- pass the new strategy here
        client_resources=client_resources,
    )


if __name__ == "__main__":
    main()
