from torchvision.transforms import Compose, Normalize, ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset

def load_datasets(cfg, partition_id: int, partitions = None):

    if partitions == None:
        partitions = cfg.num_dataset_partitions
        
    partitioner = IidPartitioner(num_partitions=partitions)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = Compose(
        [ToTensor(),
          Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    #Apply Tranforms
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train_test["train"], batch_size=cfg.BATCH_SIZE, shuffle=True )
    valloader = DataLoader(partition_train_test["test"], batch_size=cfg.BATCH_SIZE)

    # Use test split for centralized evaluation.
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=cfg.BATCH_SIZE)

    return trainloader, valloader, testloader




