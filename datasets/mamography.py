from .base_datamodule import BaseDataModule
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import random_split

class CustomDataModule(BaseDataModule):
    def __init__(self, **params):
        super(CustomDataModule, self).__init__(**params)

    def setup(self, stage: str):
        # Load the full dataset with training transforms
        dataset = ImageFolder(
            root=self.data_path,
            transform=self.train_transforms,
        )
        # Load the full dataset again with test transforms
        dataset_test = ImageFolder(
            root=self.data_path,
            transform=self.test_transforms,
        )

        n = len(dataset)  # total number of examples
        n_test = int(0.2 * n)  # take ~20% for test
        n_train = n - n_test  # the rest for training

        # Ensure the same split indices are used for both datasets
        train_indices, test_indices = random_split(range(n), [n_train, n_test])

        self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(dataset_test, test_indices)