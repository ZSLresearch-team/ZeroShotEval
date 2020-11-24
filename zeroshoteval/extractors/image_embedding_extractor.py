"""
"""
from torch import Tensor, cuda, nn
from torchvision import transforms
from torchvision.models.resnet import resnet101

from logging import getLogger
from typing import Optional

from zeroshoteval.utils.types import EmbeddingObject, ExtractorType, ImageObject
from zeroshoteval.extractors.extractor_base import EmbeddingExtractor
from zeroshoteval.utils.exceptions import ExtractorTypeError


class ImageEmbeddingExtractor(EmbeddingExtractor):

    # you can add other models from torchvision.models
    AVAILABLE_EXTRACTOR_TYPES = {"resnet101": resnet101}
    DEFAULT_TRANSFORM = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(  # normalize for pretrained models
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def __init__(
        self,
        extractor_type: ExtractorType = "resnet101",
        use_gpu: bool = False,
        cuda_device_id: int = 0,
    ):

        if extractor_type not in self.AVAILABLE_EXTRACTOR_TYPES:
            raise ExtractorTypeError

        self.logger = getLogger(
            self.__class__.__name__ + f"_{self.extractor_type}"
        )
        self.gpu = use_gpu

        if use_gpu:
            self.logger.info("GPU was chosen for embedding extraction")

            if not cuda.is_available():
                self.logger.warning("CUDA is not available, will use CPU")
                self.gpu = False
            else:
                try:
                    self.cuda_device = cuda.device(cuda_device_id)
                    self.cuda_device_id = cuda_device_id
                except AssertionError as ae:
                    self.logger.error(
                        f"device with id {cuda_device_id} is not available -"
                        f"will use device with id 0",
                        exc_info=True,
                    )
                    self.cuda_device_id = 0
                    self.cuda_device = cuda.device(self.cuda_device_id)

                self.logger.info(
                    f"CUDA is available, will use {cuda.get_device_name(self.cuda_device.idx)}"
                )
        else:
            self.logger.info("CPU was chosen for embedding extraction")

        super().__init__(extractor_type=extractor_type)

        self._source_model = self.AVAILABLE_EXTRACTOR_TYPES[extractor_type](
            pretrained=True
        )
        self.model = self._get_model_without_last_layer(self._source_model)

    @staticmethod
    def _get_model_without_last_layer(model: nn.Module) -> nn.Module:
        return nn.Sequential(*(list(model.children())[:-1]))

    @property
    def embedding_size(self):
        return self.model.fc.out_features

    def extract_embedding(
        self, object: ImageObject
    ) -> Optional[EmbeddingObject]:
        try:
            x: Tensor = self.DEFAULT_TRANSFORM(object)
            if self.gpu:
                x = x.to(self.cuda_device)
            embedding: EmbeddingObject = self.model(x)
            return embedding.view((1, -1))

        except Exception as e:
            self.logger.error(e, exc_info=True)
            return None
