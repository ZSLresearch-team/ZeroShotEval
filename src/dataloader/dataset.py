import torch
from torch.utils.data import Dataset

import numpy as np


class ObjEmbeddingDataset(Dataset):
    """Object embeddings dataset"""

    def __init__(self, data, labels, split_df, object_modalities):
        """
        Args:
            data(dict): dict mapping modalities name to modalities object embeddings.
            labels: list of ground truth labels for objects.
            split_df: pandas dataframe with dataset splits
            object_modalities: list of modalities names, wich describes objects, not classes.
        """
        self.data = data
        self.labels = labels
        self.split_df = split_df

        self.modalities = data.keys()
        self.object_modalities = object_modalities
        self.class_modalities = list(set(self.modalities) - set(object_modalities))

        self.train_indices = split_df[split_df['is_train'] == 1]['obj_id'].to_numpy()
        self.test_indices = split_df[split_df['is_train'] == 0]['obj_id'].to_numpy()
        self.test_seen_indices = split_df[(split_df['is_train'] == 0) & (split_df['is_seen'] == 1)]['obj_id'].to_numpy()
        self.test_unseen_indices = split_df[split_df['is_seen'] == 0]['obj_id'].to_numpy()

        self.seen_classes = np.unique(self.labels[self.train_indices])
        self.unseen_classes = np.unique(self.labels[self.test_unseen_indices])
        self.num_seen_classes = len(self.seen_classes)
        self.num_unseen_classes = len(self.unseen_classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_label = self.labels[idx]
        sample_data = {}

        for modality_name, modality_data in self.data.items():
            if modality_name in self.object_modalities:
                sample_data[modality_name] = modality_data[idx]

            else:
                sample_data[modality_name] = modality_data[sample_label]

        return sample_data, sample_label

    def get_zsl_emb_indice(self, samples_per_modality_class, generalized=True):
        r"""
        Creates indices of dataset to create zsl embeddings dataset.

        Args:
            samples_per_modality_class(dict): dictionary mapping modality names to its samples per class
            generalized(bool): if ``True`` get indices for generalized mod, if ``False`` -
            for few-shot learning

        Returns:
            indices_obj: array with indices of object modality
            indices_class: array with indices of class modality
        """
        if generalized:
            # !replase with dict : modality_name -> indices
            indices_obj = np.array([], dtype=np.int64)
            indices_class = np.array([], dtype=np.int64)

            for modality_name, samples_per_class in samples_per_modality_class.items():
                if modality_name in self.object_modalities:
                    for label in self.seen_classes:
                        class_indices = np.intersect1d(np.sort(self.train_indices), np.where(self.labels == label))

                        class_indices = np.resize(class_indices, samples_per_class)
                        indices_obj = np.append(indices_obj, class_indices)
                else:
                    for label in self.unseen_classes:
                        class_indices = np.where(self.labels == label)

                        class_indices = np.resize(class_indices, samples_per_class)
                        indices_class = np.append(indices_class, class_indices)

        else:
            # TODO: impliment few-shot learning
            pass

        return indices_obj, indices_class