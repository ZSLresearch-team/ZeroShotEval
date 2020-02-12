"""
"""

from src.modalities_feature_extractors.image_embedding_extractor import ImageEmbeddingExtractor
from src.modalities_feature_extractors.attributes_embedding_extractor import AttributesEmbeddingExtractor

from utils.pandas_batch_iterator import PandasBatchIterator


class ModalitiesFeatureExtractor:
    def __init__(self, modalities_dict):
        self.modalities_dict = modalities_dict

    def compute_embeddings(self):
        embeddings_dict = {}
        for mod_name, data in modalities_dict.items():
            if mod_name == 'img':
                # embedding_extractor = ImageEmbeddingExtractor()  # TODO: complete parameters
                # dataset_iterator = PandasBatchIterator(dataframe=data, batch_size=)
                # embeddings_dict[mod_name] = self.compute_image_embeddings(embedding_extractor, dataset_iterator)
                pass
            elif mod_name == 'cls_attr':
                pass
            # elif mod_name == ...
            #
            else:
                raise ValueError('Unknown modality name: {}'.format(mod_name))

        return embeddings_dict

    def compute_image_embeddings(self, embedding_extractor, dataset_iterator, dataset_folder):
        pass

    def compute_attributes_embeddings(self):
        pass