import torch
from client import LiGOClient
import flwr as fl
from data import construct_dataset
from strategy import FedLiGO
import argparse
import yaml
from utils import set_seed
import wandb
import os
import time


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
        config: dict = yaml.load(f, yaml.FullLoader)

    set_seed(config["seed"])

    def client_fn(cid) -> LiGOClient:
        trainset, testset = construct_dataset(
            config["dataset"],
            config["data_root"],
            config["num_clients"],
            config["iid_degree"],
            int(cid),
        )

        # Start Flower client
        return LiGOClient(trainset, testset, DEVICE, config, idx=int(cid))

    # config log file
    log_path = os.path.join(config["log_dir"], config["exp_name"])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(
        log_path,
        time.strftime(
            "%Y-%m-%d_%H_%M_%S",
        )
        + ".log",
    )
    fl.common.logger.configure(identifier=config["exp_name"], filename=log_file)
    # init wandb
    local_ep = config["local_ep"]
    batch_size = config["batch_size"]
    iid_degree = config["iid_degree"]
    aggregation = "AGG" if config["aggregation"] else "NOAGG"

    wandb.init(
        project="FedLiGO",
        name=f"{aggregation},iid_deg{iid_degree},local_ep{local_ep},batch_size{batch_size}",
        # Track hyperparameters and run metadata
        config=config,
    )
    # start the simulation process
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config["num_clients"],
        config=fl.server.ServerConfig(num_rounds=config["epochs"]),
        strategy=FedLiGO(config),  # <-- pass the new strategy here
        client_resources=client_resources,
    )


if __name__ == "__main__":
    main()
