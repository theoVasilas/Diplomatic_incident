from flwr_datasets import FederatedDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# BATCH_SIZE = 32

def load_datasets(cfg, partition_id: int, partitions = None):

    if partitions == None:
        partitions = cfg.num_dataset_partitions
        
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": cfg.num_dataset_partitions}
    )

    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition = fds.load_partition(partition_id)
    # Divider : 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    #Apply Tranforms
    partition_train_test = partition_train_test.with_transform(apply_transforms)

    trainloader = DataLoader(partition_train_test["train"], batch_size=cfg.BATCH_SIZE, shuffle=True )
    valloader = DataLoader(partition_train_test["test"], batch_size=cfg.BATCH_SIZE)

    # Use test split for centralized evaluation.
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=cfg.BATCH_SIZE)

    return trainloader, valloader, testloader




