from typing import Callable, Union, Optional, Dict, List, Tuple
import wandb
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import flwr as fl
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from utils import (
    gen_hetro_model_args,
    set_seed,
    get_model_params,
    weighted_metrics_avg,
    log_wandb,
)
import numpy as np
from logging import INFO
from flwr.common import log
from datetime import datetime


def save_ndarrays_to_npz(base_filename: str, arrays: NDArrays):
    # Format the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the full filename with timestamp
    filename = f"{base_filename}{timestamp}.npz"
    # Create a dictionary to store arrays with unique names
    array_dict = {f"array_{i}": arr for i, arr in enumerate(arrays)}
    # Save to .npz file
    np.savez(filename, **array_dict)
    print(f"Saved arrays to {filename}")


def load_ndarrays_from_npz(file_path: str) -> NDArrays:
    with np.load(file_path) as data:
        return [data[key] for key in data.files]


class FedLiGO(fl.server.strategy.Strategy):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        # setup seed
        set_seed(self.config["seed"])
        # save arguments to member variables
        self.num_clients = self.config["num_clients"]
        self.weights = [
            Parameters(tensors=[], tensor_type="") for _ in range(self.num_clients)
        ]
        self.client_configs = [
            gen_hetro_model_args(self.config) for _ in range(self.num_clients)
        ]
        # log
        log(
            INFO,
            "Strategy init successful, the client's model config is: {}".format(
                self.client_configs
            ),
        )

    def __repr__(self) -> str:
        return "FedLiGO Strategy"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        # We apply hetrogeneous parameters to all clients.
        # So there is no need to perform unified parameter assignment.
        pass

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        fit_configuration : List[Tuple[ClientProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """
        client_manager.wait_for(self.num_clients)
        # Sample clients
        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients,
        )

        # Create custom configs for FedLe

        # 1. At first we begin small model training and ligo operator training.
        if self.config["small_model_training"] and server_round == 1:
            # send small model training instructions to each client.
            fit_configurations = []
            for idx, client in enumerate(clients):
                fit_configurations.append(
                    (
                        client,
                        FitIns(
                            parameters,
                            {
                                "small_model_idx": self.client_configs[idx],
                                "small_model_training": True,
                            },
                        ),
                    )
                )
            log(
                INFO,
                "Server: round {} begin. Train small model".format(server_round),
            )
        # 2. After the small model trianing process,
        # we begin to train and aggregate the ligo operators.
        else:
            if self.config["aggregation"]:
                fit_configurations = []
                for idx, client in enumerate(clients):
                    fit_configurations.append(
                        (
                            client,
                            FitIns(
                                parameters,
                                {
                                    "small_model_idx": self.client_configs[idx],
                                    "small_model_training": False,
                                },
                            ),
                        )
                    )
                log(
                    INFO,
                    "Server: round {} begin. Enable parameter aggregation".format(
                        server_round
                    ),
                )
            else:
                fit_configurations = []
                for idx, client in enumerate(clients):
                    fit_configurations.append(
                        (
                            client,
                            FitIns(
                                self.weights[idx],
                                {
                                    "small_model_idx": self.client_configs[idx],
                                    "small_model_training": False,
                                },
                            ),
                        )
                    )
                log(
                    INFO,
                    "Server: round {} begin. Disable parameter aggregation".format(
                        server_round
                    ),
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.

        Returns
        -------
        parameters : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        # perform weighted parameter aggregation
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        # record the weight_results for no_aggregation clients
        for client_proxy, fit_res in results:
            self.weights[int(client_proxy.cid)] = fit_res.parameters

        # get the aggregated training results
        metrics = [(fit_res.metrics, fit_res.num_examples) for _, fit_res in results]
        aggregated_params = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_params)
        metrics_aggregated = weighted_metrics_avg(metrics)
        # log
        log(
            INFO,
            "Server: round {} finished, the aggregated metrics are: {}".format(
                server_round, metrics_aggregated
            ),
        )
        log_wandb([fit_res for _, fit_res in results])

        # save parameters
        if server_round == self.config["epochs"]:
            save_ndarrays_to_npz("global_model_", aggregated_params)
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        evaluate_configuration : List[Tuple[ClientProxy, EvaluateIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `EvaluateIns` for this particular `ClientProxy`. If a particular
            `ClientProxy` is not included in this list, it means that this
            `ClientProxy` will not participate in the next round of federated
            evaluation.
        """

        # Sample clients
        clients = client_manager.sample(
            num_clients=client_manager.num_available(),
            min_num_clients=client_manager.num_available(),
        )
        # set the eval instructions
        eval_ins = []
        for idx, client in enumerate(clients):
            eval_ins.append(
                (
                    client,
                    EvaluateIns(
                        parameters if self.config["aggregation"] else self.weights[idx],
                        {},
                    ),
                )
            )

        return eval_ins

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the
            previously selected and configured clients. Each pair of
            `(ClientProxy, FitRes` constitutes a successful update from one of the
            previously selected clients. Not that not all previously selected
            clients are necessarily included in this list: a client might drop out
            and not submit a result. For each client that did not submit an update,
            there should be an `Exception` in `failures`.
        failures : List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
            Exceptions that occurred while the server was waiting for client updates.

        Returns
        -------
        aggregation_result : Optional[float]
            The aggregated evaluation result. Aggregation typically uses some variant
            of a weighted average.
        """

        if not results:
            return None, {}
        # perform weighted loss avg and weighted matrix avg
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics = [(fit_res.metrics, fit_res.num_examples) for _, fit_res in results]

        metrics_aggregated = weighted_metrics_avg(metrics)
        # log
        log(
            INFO,
            "Server: The evaluation metrics are: {}".format(metrics_aggregated),
        )
        log_wandb([fit_res for _, fit_res in results])
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current model parameters.

        This function can be used to perform centralized (i.e., server-side) evaluation
        of model parameters.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters: Parameters
            The current (global) model parameters.

        Returns
        -------
        evaluation_result : Optional[Tuple[float, Dict[str, Scalar]]]
            The evaluation result, usually a Tuple containing loss and a
            dictionary containing task-specific metrics (e.g., accuracy).
        """

        # Since we only aggregate the LiGO operator in the server side, we can not evaluate the performance of the aggregated global model.
        pass
