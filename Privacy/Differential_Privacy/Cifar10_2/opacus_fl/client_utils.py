from typing import Dict, List, Optional, Tuple
from collections import OrderedDict, Counter

from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

from opacus import PrivacyEngine
from opacus.accountants.analysis import gdp

import numpy as np
import torch
import time

from data_utils import load_datasets
from train_utils import Net, train_DP,train_DP_with_epsilon, train, test
from extra_utils import set_seed

import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="opacus")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # replace the parameters
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(NumPyClient):
    def __init__(self,
                 pid,
                 net,
                 trainloader,
                 valloader,
                 cfg,
                 ):

        self.pid = pid  # partition ID of a client
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

        self.clipping = cfg.clipping
        self.Dp_enabel = cfg.Dp_enabel
        self.Dp_e_enabel = cfg.Dp_e_enabel
        self.batch_size = cfg.BATCH_SIZE

        self.seed = cfg.SEED,

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        warnings.filterwarnings("ignore", category=UserWarning)
    
    def get_parameters(self, config):
        print(f"[Client {self.pid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        
        start_time = time.time()

        # Read configuration values
        LEARNING_RATE = config["learning rate"]
        NOISE_MULTIPLIER = config["noise_multiplier"]
        MAX_GRAD_NORM = config["max_grad_norm"]
        TARGET_EPSILON = config["target_epsilon"]
        TARGET_DELTA = config["target_delta"]
        LOCAL_EPOCHS = config["local_epochs"]
        SEED = config["SEED"]
                      
        set_seed(SEED)
        # print("\n\n {seed} \n\n")

        # Set model parameters
        model = self.net
        set_parameters(self.net, parameters)
        # Setup optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

        if (self.Dp_enabel == True) and (self.Dp_e_enabel == False):
            # privacy_engine = PrivacyEngine(secure_mode=True)
            # privacy_engine = PrivacyEngine(secure_mode=False) # to make it deterministic
            privacy_engine = PrivacyEngine(secure_mode=False, accountant="gdp")
            privacy_engine.seed = SEED

            new_model, new_optimizer, new_train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=self.trainloader,

                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
            )

            avg_trainloss, epsilon, max_norm_before_clipping, grand_70_pec = train_DP(
                        new_model,
                        new_train_loader,
                        privacy_engine,
                        new_optimizer,

                        TARGET_DELTA,

                        self.device, 
                        LOCAL_EPOCHS)
            
            if epsilon is not None:
                print(f"Epsilon = {epsilon:.2f} for target (noise_multiplier: {NOISE_MULTIPLIER}, delta: {TARGET_DELTA}) ")
            else:
                print("Epsilon value not available.")

            print(f" max_norm {max_norm_before_clipping}, grand_70_pec {grand_70_pec}\n")

        elif (self.Dp_enabel and self.Dp_e_enabel) == True:
            # privacy_engine = PrivacyEngine(secure_mode=True)
            # privacy_engine = PrivacyEngine(secure_mode=False, accountant="gdp")
            privacy_engine = PrivacyEngine(secure_mode=False)
            privacy_engine.seed = SEED

            new_model, new_optimizer, new_train_loader = privacy_engine.make_private_with_epsilon( 
                module=self.net,
                optimizer=optimizer,
                data_loader=self.trainloader,

                target_epsilon=TARGET_EPSILON,
                target_delta=TARGET_DELTA,

                epochs=LOCAL_EPOCHS,
                max_grad_norm=MAX_GRAD_NORM,
                clipping=self.clipping, # (flat or per_layer or adaptive)
            )
            print(f"Using sigma={new_optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

            sample_rate = min(self.batch_size / len(new_train_loader), 1) 

            avg_trainloss, noise_multiplier, max_norm_before_clipping, grand_70_pec = train_DP_with_epsilon(
                        net=new_model,
                        train_loader=new_train_loader,
                        privacy_engine=privacy_engine,
                        target_epsilon=TARGET_EPSILON,
                        target_delta=TARGET_DELTA,
                        sample_rate=sample_rate,
                        optimizer=new_optimizer,
                        device=self.device,
                        epochs=LOCAL_EPOCHS
                        )
            
            epsilon = TARGET_EPSILON
            if noise_multiplier is not None:
                print(f"noise_multiplier = {noise_multiplier:.2f} for target (epsilon: {TARGET_EPSILON}, delta: {TARGET_DELTA})")
            else:
                print("noise_multiplier value not available.")

            print(f" max_norm {max_norm_before_clipping}, grand_70_pec {grand_70_pec}\n")

        else:
            new_model = self.net
            avg_trainloss, max_norm_before_clipping, grand_70_pec = train(new_model,
                            self.trainloader,
                            optimizer,
                            self.device,
                            epochs=LOCAL_EPOCHS)
            epsilon = 0 

        end_time = time.time()
        elapsed_time = end_time - start_time

        results = {
        "train_loss": avg_trainloss,
        "noise_multiplier": NOISE_MULTIPLIER,
        "epsilon": epsilon, 
        "max_norm_before_clipping": max_norm_before_clipping,
        "grand_70_pec": grand_70_pec,
        "elapsed_time": elapsed_time,
        }

        return get_parameters(new_model), len(self.trainloader), results

    def evaluate(self, parameters, config):

        # current_round = config["server_round"]
        print(f"[Client {self.pid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)

        loss, accuracy = test(self.net, self.valloader)

        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "loss": float(loss)}


def create_client_app(cfg):
    def client_fn(context: Context) -> Client:
        net = Net()
        partition_id = context.node_config["partition-id"]
        # num_partitions = context.node_config["num-partitions"]
        trainloader, valloader, _= load_datasets(cfg, partition_id)
        return FlowerClient(partition_id, net, trainloader, valloader, cfg).to_client()
    
    client_app = ClientApp(client_fn=client_fn)
    return client_app


def get_fit_config(cfg):

    def fit_config(server_round: int):
        config = {
            "local_epochs"      : cfg.local_epochs ,
            "learning rate"     : cfg.learning_rate,
            "noise_multiplier"  : cfg.noise_multiplier ,
            "max_grad_norm"     : cfg.max_grad_norm  ,
            "target_epsilon"    : cfg.target_epsilon  ,
            "target_delta"      : cfg.target_delta  ,
            "SEED"              : cfg.SEED
        }
        return config

    return fit_config


def evaluate_config(server_round: int):
    config = {
        "current_round": server_round,
    }
    return config


