"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import logging
from typing import List, Tuple

from opacus_fl.task import Net, get_weights

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from typing import List, Tuple

from flwr.server import (
    Driver,
    LegacyContext,
    ServerApp,
    ServerConfig,
)
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow

from codecarbon import track_emissions


# Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
logging.getLogger("flwr").setLevel(logging.INFO)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

app = ServerApp()

@app.main()
@track_emissions(project_name="Dif_Pr_Opacus")
def main(driver: Driver, context: Context) -> None:

    # Initialize global model
    model_weights = get_weights(Net())
    parameters = ndarrays_to_parameters(model_weights)

    # Note: The fraction_fit value is configured based on the DP hyperparameter `num-sampled-clients`.
    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=5,
        # fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn = weighted_average,
        initial_parameters=parameters,
    )


    num_rounds = context.run_config["num-server-rounds"]

    # Construct the LegacyContext
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Create the train/evaluate workflow
    workflow = DefaultWorkflow()

    # Execute
    workflow(driver, context)