print("Importing server.py")

import flwr as fl
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import FitRes, EvaluateRes, Parameters, Scalar
import logging


def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0}


class MovieLensServer(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(
            # min_fit_clients=10,
            # min_evaluate_clients=10,
            # min_available_clients=10,
            *args,
            **kwargs
        )
        self.round_accuracies = []
        self.client_spearman_correlations = {}
        self.client_accuracies = {}
        self.client_id_map = {}  # Map to maintain consistent numerical client IDs
        self.client_num_examples_per_round = {}  # Initialize the dictionary to store examples per client per round

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,  # Change to DEBUG for more detailed logs
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        logging.info("MovieLensServer initialized.")

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        aggregated = super().aggregate_fit(rnd, results, failures)
        logging.info(f"Round {rnd}: Aggregated fit results.")
        return aggregated

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[BaseException]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if not results:
            logging.warning(f"Round {rnd}: No evaluation results received.")
            return None

        accuracies = []
        for client, r in results:
            if client.cid not in self.client_id_map:
                self.client_id_map[client.cid] = f"Client {len(self.client_id_map)}"
                logging.info(f"Registered new client: {self.client_id_map[client.cid]} with CID: {client.cid}")
            client_id = self.client_id_map[client.cid]

            accuracy = r.metrics.get("accuracy", 0.0)
            spearman_corr = r.metrics.get("spearman_correlation", 0.0)
            num_examples = r.num_examples

            # Accumulate weighted accuracies
            accuracies.append(accuracy * num_examples)

            # Record the number of examples per client per round
            if client_id not in self.client_num_examples_per_round:
                self.client_num_examples_per_round[client_id] = []
            self.client_num_examples_per_round[client_id].append(num_examples)
            logging.debug(f"Round {rnd}, {client_id}: Accuracy={accuracy}, Spearman={spearman_corr}, Examples={num_examples}")

            # Record Spearman correlations
            if client_id not in self.client_spearman_correlations:
                self.client_spearman_correlations[client_id] = []
            self.client_spearman_correlations[client_id].append(spearman_corr)

            # Record client accuracies
            if client_id not in self.client_accuracies:
                self.client_accuracies[client_id] = []
            self.client_accuracies[client_id].append(accuracy)

        # Compute aggregated accuracy using weighted average
        total_accuracy = sum(accuracies)
        total_examples = sum([num_examples for num_examples in [r.num_examples for _, r in results]])
        aggregated_accuracy = total_accuracy / total_examples if total_examples > 0 else 0.0
        self.round_accuracies.append(aggregated_accuracy)
        logging.info(f"Round {rnd}: Aggregated Accuracy = {aggregated_accuracy}")

        return 0.0, {"accuracy": aggregated_accuracy}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        config: Dict[str, Scalar] = None
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if config is None:
            config = {}
        try:
            result = super().evaluate(server_round, parameters, config)
            logging.info(f"Round {server_round}: Evaluation completed.")
        except TypeError:
            try:
                result = super().evaluate(server_round, parameters)
                logging.info(f"Round {server_round}: Evaluation completed (TypeError handled).")
            except NotImplementedError:
                logging.error(f"Round {server_round}: Evaluation not implemented.")
                return None

        return result
