import torch
from client import LiGOClient
import flwr as fl
from data import construct_dataset
from strategy import FedLiGO
import argparse
import yaml
from utils import set_seed


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
        client_resources = {"num_gpus": 1, "num_cpus": 10}

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
        return LiGOClient(trainset, testset, DEVICE, None)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config["num_clients"],
        config=fl.server.ServerConfig(num_rounds=config["epochs"]),
        strategy=FedLiGO(config),  # <-- pass the new strategy here
        client_resources=client_resources,
    )


if __name__ == "__main__":
    main()
