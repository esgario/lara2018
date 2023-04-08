import math
from abc import ABC, abstractmethod


class ModelTraining(ABC):
    """Abstract class for training a model."""

    def __init__(self):
        pass

    @abstractmethod
    def train(self, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def validation(self, **kwargs):
        """Validate the model."""
        pass

    @abstractmethod
    def run_training(self, **kwargs):
        """Run the training."""
        pass

    @abstractmethod
    def run_test(self, **kwargs):
        """Run the test."""
        pass

    def adjust_learning_rate(self, optimizer, epoch, opt):
        """Adjust the learning rate according to the epoch.

        Args:
            optimizer (torch.optim): The optimizer.
            epoch (int): The current epoch.
            opt (argparse.Namespace): The options.

        Returns:
            torch.optim: The optimizer with the adjusted learning rate.
        """
        if opt.optimizer == "sgd":
            lr_values = [0.01, 0.005, 0.001, 0.0005, 0.0001]
            step = round(opt.epochs / 5)

            idx = min(math.floor(epoch / step), len(lr_values))
            learning_rate = lr_values[idx]

            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

        return optimizer
