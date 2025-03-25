"""fl_dp_sa: Flower Example using Differential Privacy and Secure Aggregation."""

from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple
import json
import toml # type: ignore
import os
import shutil


from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import (
    Driver,
    LegacyContext,
    ServerApp,
    ServerConfig,
)
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from Flower_sa.task import Net, get_weights, load_data, test_2, set_weights
from Flower_sa.extra_utils import set_seed # type: ignore
from codecarbon import track_emissions, EmissionsTracker 

from flwr.common import Metrics, Context, ndarrays_to_parameters, NDArrays, Scalar
from typing import Dict, List, Optional, Tuple
import torch


def get_clients_fit_fn(output_dir):

    output_file = "clients_evaluation_history.json"
    output_path  = os.path.join(output_dir, output_file)

    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:

        # print("Metrics:", metrics)
        examples = [num_examples for num_examples, _ in metrics]

        train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
        train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]

        val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
        val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

        results = {
            "train_loss"    : sum(train_losses) / sum(examples),
            "train_accuracy": sum(train_accuracies) / sum(examples),
            "val_loss"      : sum(val_losses) / sum(examples),
            "val_accuracy"  : sum(val_accuracies) / sum(examples),
        }
    
        with open(output_path , "a") as f:  # Append to file
            f.write(json.dumps(results) + "\n")

        return results
    
    return weighted_average


def get_server_evaluate_fn(output_dir):

    output_file = "server_evaluation_history.json"
    output_path  = os.path.join(output_dir, output_file)

    def My_evaluate(
            server_round: int,
            parameters: NDArrays,
            config: Dict[str, Scalar],
        ):

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = Net().to(DEVICE)

        _, _, testloader = load_data(partition_id=0,num_partitions=1)
        set_weights(net, parameters)
        loss, acc, prec, rec, f1 = test_2(net, testloader)

        print(f"\t Server-side evaluation loss {loss} / accuracy {acc}")

        # Save metrics to a file
        eval_entry = {
            "round": server_round,
            "loss": loss,
            "accuracy": acc,
            "preccision" : prec,
            "recall" : rec,
            "f1": f1,
        }

        with open(output_path , "a") as f:  # Append to file
            f.write(json.dumps(eval_entry) + "\n")

        return loss, {"loss": loss, "accuracy": acc}
    
    return My_evaluate

app = ServerApp()


# @track_emissions(project_name="SecAggr")
@app.main()
def main(driver: Driver, context: Context) -> None:

    #=========== PULL configurations ===================
    base_path = project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # print(base_path)
    config = toml.load(os.path.join(base_path, "pyproject.toml") )

    SEED = context.run_config["SEED"]
    set_seed(SEED)

    NUM_ROUNDS = context.run_config["num-rounds"]

    # Parameters for the SecAgg+ protocol
    NUM_SHARES                  = context.run_config["num-shares"]
    RECONSTRUCTION_THRESHOLD    = context.run_config["reconstruction-threshold"]
    CLIPPING_RANGE              = context.run_config["clipping-range"]
    QUANTIZATION_RANGE          = context.run_config["quantization-range"]

    NUM_SUPERNODES = config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]
    print("\n==== Experiment Configuration ====")
    print(f"SEED: {SEED}")
    print("\n-- SecAgg+ Parameters --")
    print(f"NUM_SHARES: {NUM_SHARES}")
    print(f"RECONSTRUCTION_THRESHOLD: {RECONSTRUCTION_THRESHOLD}")
    print(f"CLIPPING_RANGE: {CLIPPING_RANGE}")
    print(f"QUANTIZATION_RANGE: {QUANTIZATION_RANGE}")
    print("\n-- Flower Federation Settings --")
    print(f"NUM_SUPERNODES: {NUM_SUPERNODES}")
    print(f"NUM_ROUNDS: {NUM_ROUNDS}")
    print("===================================")



    lock_file = "/tmp/.codecarbon.lock"

    # Check if lock file exists and remove it
    if os.path.exists(lock_file):
        os.remove(lock_file)
    tracker = EmissionsTracker(save_to_file=False) #log_level="error", save_to_file=False
    #================= Create output Folder ==========================
    os.makedirs("outputs", exist_ok=True)

    print(f"\n { tracker._conf['cpu_model'] }")
    if tracker._conf['cpu_model']=="AMD Ryzen 5 5600G with Radeon Graphics" :
        machine_output_dir = "local"
    else :
        machine_output_dir = "Remote"

    base_output_dir = os.path.join("outputs", machine_output_dir)

    path = f"C_{NUM_SUPERNODES}_R{NUM_ROUNDS}_NSh{NUM_SHARES}_Recth{RECONSTRUCTION_THRESHOLD}_ClpR{CLIPPING_RANGE}_Qr{QUANTIZATION_RANGE}/s{SEED}"
    output_dir = os.path.join(base_output_dir, path)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory is created

    print(f"Output directory: {output_dir}")
    #========================================================================


    # Initialize global model
    # Reset the model
    model = Net()
    _, _, testloader = load_data(partition_id=0,num_partitions=1)
    loss, acc, prec, rec, f1 = test_2(model, testloader)
    print(f"\n model is reset.. acc= {acc} \n")

    model_weights = get_weights(model)
    parameters = ndarrays_to_parameters(model_weights)

    # Note: The fraction_fit value is configured based on the DP hyperparameter `num-sampled-clients`.
    strategy = FedAvg(
        fraction_fit = 1,
        fraction_evaluate = 1,
        evaluate_fn = get_server_evaluate_fn(output_dir),
        # on_fit_config_fn =,
        # on_evaluate_config_fn=,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn = get_clients_fit_fn(output_dir),
        # evaluate_metrics_aggregation_fn = ,
    )


    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # Create the train/evaluate workflow
    workflow = DefaultWorkflow(
        fit_workflow=SecAggPlusWorkflow(
            num_shares                  = NUM_SHARES               ,
            reconstruction_threshold    = RECONSTRUCTION_THRESHOLD ,
            clipping_range              = CLIPPING_RANGE           ,
            quantization_range          = QUANTIZATION_RANGE       ,
 
        )
    )

    # ========EmissionsTracker===============
    tracker.start()
    # =======================================

    # Execute
    workflow(driver, context)

    # =======================================
    tracker.stop()
    emissions_data = tracker.final_emissions_data.__dict__
    print(f"\n duration : {emissions_data['duration']} \n") 
    # =======================================

    all_results = {
            "num_supernodes"            : NUM_SUPERNODES,
            "num_shares"                : NUM_SHARES  ,            
            "reconstruction_threshold"  : RECONSTRUCTION_THRESHOLD,
            "cliping_range"             : CLIPPING_RANGE   ,       
            "quantization_range"        : QUANTIZATION_RANGE  ,    
            "num_rounds"                : NUM_ROUNDS,
            "duration"                  : emissions_data['duration'],
            "emissions"                 : emissions_data['emissions'],
        }
    # print(all_results)



    # ============ SAVE RESULTS =============================

    CodeCarbon_file = os.path.join(output_dir, "CodeCarbon_data.json")
    with open(CodeCarbon_file, "w") as f:  
        json.dump(emissions_data, f)
        f.write("\n")  # Ensure each entry is on a new line
    
    output_file = os.path.join(output_dir, "Sim_results.json")
    with open(output_file, "w") as f:  
        json.dump(all_results, f)
        f.write("\n")  # Ensure each entry is on a new line


    