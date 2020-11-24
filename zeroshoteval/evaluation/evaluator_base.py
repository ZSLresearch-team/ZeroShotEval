import logging


class EvaluatorBase:

    def __init__(self, model, data_loader) -> None:

        # We set the model to evaluation mode in the evaluator.
        model.eval()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)

    def evaluate(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info("Starting evaluation...")

        raise NotImplementedError
