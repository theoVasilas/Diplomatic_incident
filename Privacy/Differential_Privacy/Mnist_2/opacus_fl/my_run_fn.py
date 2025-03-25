from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context, ndarrays_to_parameters, NDArrays, Scalar
from flwr.simulation import run_simulation
from flwr.server.strategy import FedAvg

from typing import Dict, List, Optional, Tuple
import logging
import torch

from data_utils import load_datasets
from client_utils import get_parameters, set_parameters, FlowerClient
from client_utils import get_fit_config, create_client_app
from server_utils import create_server_app

from train_utils import test, test_2, Net
from extra_utils import set_seed, save_results
from extra_utils import plot_aggregation_history, plot_evaluation_history

import json
import os

def get_eval_weighted_average(cfg, ):

    output_file = "clients_eval_aggregation_history.json"

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        losses = [num_examples * m.get("loss", 0.0) for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        
        accuracy = sum(accuracies) / sum(examples)
        loss = sum(losses) / sum(examples)

        # Save metrics to history
        Aggregations_entry = {
            "loss": loss,
            "accuracy": accuracy
        }

        with open(output_file, "a") as f:  # Append to file
            f.write(json.dumps(Aggregations_entry) + "\n")

        print(f"My_weighted_average- accuracy: {accuracy} loss: {loss} ")

        return {"clients_accuracies": accuracy, "clients_losses": loss}
    
    return weighted_average


def get_fit_weighted_average(cfg, ):

    output_file = "clients_fit_aggregation_history.json"

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        
        # print(f"\n\n metrics : {metrics}")

        examples = [num_examples for num_examples, _ in metrics]

        losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
        loss = sum(losses) / sum(examples)
        
        if cfg.Dp_e_enabel:
            # Calculate noise multiplier dynamically
            noise_multipliers = [num_examples * m["noise_multiplier"] for num_examples, m in metrics]
            noise_multiplier = sum(noise_multipliers) / sum(examples)
            epsilon = cfg.target_epsilon
        else:
            # Calculate epsilon dynamically
            epsilons = [num_examples * m["epsilon"] for num_examples, m in metrics]
            epsilon = sum(epsilons) / sum(examples)
            noise_multiplier = cfg.noise_multiplier


        all_max_norms = [m["max_norm_before_clipping"] for _, m in metrics] 
        max_norm = max(all_max_norms)

        # Save metrics to history
        Aggregations_entry = (
            {
            "loss": loss,
            "epsilon": epsilon,
            "noise_multiplier" : noise_multiplier,
            "max_norm" : max_norm,
            },
            all_max_norms,
        )

        with open(output_file, "a") as f:  # Append to file
            f.write(json.dumps(Aggregations_entry) + "\n")

        print(f"\n fit_weighted_average- noise_multiplier: {noise_multiplier} epsilon: {epsilon} loss: {loss} ")

        return {"clients_epsilons": epsilon, "clients_losses": loss}
    
    return weighted_average


def get_My_evaluate_fn(cfg, output_dir="output"):

    output_file = "server_evaluation_history.json"

    def My_evaluate(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = Net().to(DEVICE)

        _, _, testloader = load_datasets(cfg, 0, partitions= 1)
        set_parameters(net, parameters)  # Update model with the latest parameters
        loss, acc, prec, rec, f1 = test_2(net, testloader)

        print(f"Server-side evaluation loss {loss} / accuracy {acc}")

        # Save metrics to a file
        eval_entry = {
            "round": server_round,
            "loss": loss,
            "accuracy": acc,
            "preccision" : prec,
            "recall" : rec,
            "f1": f1,
        }

        with open(output_file, "a") as f:  # Append to file
            f.write(json.dumps(eval_entry) + "\n")

        return loss, {"accuracy": acc}

    return My_evaluate


def run_my_sim(cfg, ):

    # Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
    logging.getLogger("flwr").setLevel(logging.INFO)

    set_seed(cfg.SEED)
    
    # Reset the model
    model = Net()
    _, valloader, _ = load_datasets(cfg, 0, partitions = 1)
    _, accuracy = test(model, valloader)
    print(f"\n model is reset.. acc= {accuracy} \n")
    
    model_weights = get_parameters(model)
    parameters = ndarrays_to_parameters(model_weights)


    strategy = FedAvg(
        fraction_fit  = cfg.fraction_fit,
        fraction_evaluate = 1,
        # min_fit_clients  =  5,
        # min_evaluate_clients  = 2,
        # min_available_clients = NUM_CLIENTS, # use all
        evaluate_fn = get_My_evaluate_fn(cfg, ), # function used for validation dy server
        on_fit_config_fn  = get_fit_config(cfg),  # Pass the fit_config function
        # on_evaluate_config_fn = evaluate_config,
        # accept_failures,
        initial_parameters  = parameters,
        fit_metrics_aggregation_fn = get_fit_weighted_average(cfg,) ,
        evaluate_metrics_aggregation_fn = get_eval_weighted_average(cfg,),
        # inplace,
    )

    client_app = create_client_app(cfg)

    server = create_server_app(cfg, strategy)

    # Backend configuration
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # Run the simulation if the flag is enabled
    run_simulation(
        server_app=server,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
    )

