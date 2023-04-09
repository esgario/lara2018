import math
from abc import ABC, abstractmethod


class ModelTraining(ABC):
    """Abstract class for training a model."""

    def __init__(self):
        pass

    @abstractmethod
    def train(self, loader, model, criterion, optimizer, data_augmentation=None):
        """Train the model."""
        pass

    @abstractmethod
    def validation(self, loader, model, criterion):
        """Validate the model."""
        pass

    @abstractmethod
    def run_training(self, *args, **kwargs):
        """Run the training."""
        pass

    @abstractmethod
    def run_test(self, *args, **kwargs):
        """Run the test."""
        pass

    def adjust_learning_rate(self, optimizer, optim_name, epoch, epochs):
        """Adjust the learning rate according to the epoch.

        Args:
            optimizer (torch.optim): The optimizer.
            optim_name (str): The name of the optimizer.
            epoch (int): The current epoch.
            epochs (int): The total number of epochs.

        Returns:
            torch.optim: The optimizer with the adjusted learning rate.
        """
        if optim_name == "sgd":
            lr_values = [0.01, 0.005, 0.001, 0.0005, 0.0001]
            step = round(epochs / 5)

            idx = min(math.floor(epoch / step), len(lr_values))
            learning_rate = lr_values[idx]

            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        return optimizer
