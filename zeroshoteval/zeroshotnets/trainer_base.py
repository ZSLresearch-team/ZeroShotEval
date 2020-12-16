import logging
import time

from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader


class TrainerBase:
    """
    Base class for iterative trainer for the most common type of task.

    It assumes that training is an iterative process and every step, you:
    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    """

    def __init__(self, model: Module, data_loader: DataLoader, optimizer: Optimizer) -> None:
        """
        Args:
            model: Torch nn.Module - takes a data from data_loader and returns a
                dict of losses.
            data_loader: An iterable - contains data to be used to call model.
            optimizer: Torch optimizer.
        """

        # NOTE: One can easily define any kind of train hooks here.

        # TODO: define simple hooks, e.g. save_model

        # We set the model to training mode in the trainer.
        # However it's valid to train a model that's in eval mode.
        # If you want your model (or a submodule of it) to behave
        # like evaluation during training, you can overwrite its train() method.
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def train(self, max_iter: int) -> None:
        """ TODO: write docstring
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training...")
        start_time: float = time.perf_counter()

        self.current_iter: int = 0
        self.max_iter: int = max_iter

        while self.current_iter < self.max_iter:
            self.run_step()
            self.current_iter += 1

        end_time: float = time.perf_counter()
        logger.info(f"Time elapsed for training: {start_time - end_time} seconds.")

    def run_step(self):
        raise NotImplementedError
