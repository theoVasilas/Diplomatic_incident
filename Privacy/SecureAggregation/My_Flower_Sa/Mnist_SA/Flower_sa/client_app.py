"""flower_sa: Flower Example using Differential Privacy and Secure Aggregation."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.client.mod import fixedclipping_mod, secaggplus_mod

from Flower_sa.task import Net, get_weights, load_data, set_weights, test, train
from Flower_sa.extra_utils import set_seed # type: ignore

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            epochs=1,
            device=self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    SEED = context.run_config["SEED"]
    set_seed(SEED)
    net = Net()

    trainloader, valloader, testloader = load_data(
        partition_id=partition_id, 
        num_partitions=context.node_config["num-partitions"]
    )

    return FlowerClient(net, trainloader, valloader, testloader).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
        # fixedclipping_mod,
    ],
)