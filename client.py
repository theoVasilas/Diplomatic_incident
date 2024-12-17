from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate

from model import test, train


class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient."""

    def __init__(self, trainloader, valloader, model_cfg, optim_cfg) -> None:
        super().__init__()
        # self.pid = pid  # partition ID of a client
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = instantiate(model_cfg)

        params = self.model.parameters()
        self.optim = instantiate(optim_cfg)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        optimizer = self.optim(params=self.model.parameters())
        # print(f"\n\n {optimizer} \n\n")

        local_epochs = config["local_epochs"]

        train(self.model, self.trainloader, optimizer, local_epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}


def generate_client_fn(trainloaders, valloaders, model_cfg, optim_cfg):
    """Return a function to construct a FlowerClient."""

    def client_fn(cid: str):

        # partition_id = context.node_config["partition-id"]

        return FlowerClient(
            # pid = partition_id
            trainloader = trainloaders[int(cid)],
            valloader   = valloaders[int(cid)],
            model_cfg   = model_cfg,
            optim_cfg   = optim_cfg,
        ).to_client()

    return client_fn