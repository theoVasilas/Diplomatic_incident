import hydra 
from omegaconf import  DictConfig, OmegaConf
from dataset import prepare_dataloaders
from client import generate_client_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)

def main(cfg: DictConfig):
    ## 1. config
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepere Dataset
    trainloaders, validationloaders, testloaders = prepare_dataloaders(cfg.num_clients,
                                                                       cfg.batch_size)

    print(len(trainloaders), len(trainloaders[0].dataset))

    ## 3. Define your client
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_callsses)

if __name__ == "__main__":

    main()