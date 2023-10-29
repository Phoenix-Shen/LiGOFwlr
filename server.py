import argparse
import flwr as fl
import warnings
from strategy import FedLiGO
import yaml
from utils import set_seed
import os
import time

warnings.filterwarnings("ignore")


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--cfg_path",
        type=str,
        required=True,
        help="Config_path of the client, should be consistent across all clients.",
    )

    args = parser.parse_args()

    # Load config file.
    with open(args.cfg_path, "r") as f:
        fedligo_cfg = yaml.load(f, Loader=yaml.FullLoader)
    set_seed(fedligo_cfg["seed"])

    # Create strategy
    strategy = FedLiGO(fedligo_cfg)
    # config log file
    log_path = os.path.join(fedligo_cfg["log_dir"], fedligo_cfg["exp_name"])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(
        log_path,
        time.strftime(
            "%Y-%m-%d_%H_%M_%S",
        )
        + ".log",
    )
    fl.common.logger.configure(identifier=fedligo_cfg["exp_name"], filename=log_file)
    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=fedligo_cfg["epochs"]),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
