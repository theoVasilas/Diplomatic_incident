import torch
import numpy as np  
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from torchvision import transforms
from pathlib import Path

def load_datasets(cfg, partition_id: int, partitions = None):

    if partitions == None:
        partitions = cfg.num_dataset_partitions

    fds = FederatedDataset(
        dataset=cfg.dataset,
        partitioners={"train": partitions}
    )

    # mean = cfg["normalize"]["mean"]
    # std = cfg["normalize"]["std"]

    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081)
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition = fds.load_partition(partition_id)
    # Divider : 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    #Apply Tranforms
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    g = torch.Generator()
    g.manual_seed(cfg.SEED)

    # Create DataLoader
    trainloader = DataLoader(
        partition_train_test["train"], 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0, 
        generator=g  # Pass the generator
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=cfg.BATCH_SIZE)

    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=cfg.BATCH_SIZE)

    return trainloader, valloader, testloader