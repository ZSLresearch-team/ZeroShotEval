from abc import ABCMeta
from os import PathLike

from zeroshoteval.utils.types import EmbeddingObject, ExtractorType, SourceObject


class EmbeddingExtractor(metaclass=ABCMeta):
    def __init__(self, extractor_type: ExtractorType):
        """
        Args:
            extractor_type: type of embedding extractor (resnet101 for images for example)
        """
        self.extractor_type = extractor_type

    def extract_embedding(self, object: SourceObject) -> EmbeddingObject:
        """
        Args:
            object: object for extracting embeddings
        Returns:
            embedding object in one of available format
        """
        pass

    def extract_embeddings_recursive_from_dir(
        self, dir_from: PathLike, dir_to: PathLike
    ) -> PathLike:
        """
        Recursively walk around a directory and extract embeddings for each file

        Args:
            dir_from: directory with source objects
            dir_to: directory for saving embeddings
        """
        pass
