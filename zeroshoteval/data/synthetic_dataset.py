import logging

import numpy as np
from scipy import io as sio
from sklearn import preprocessing
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GenEmbeddingDataset(Dataset):
    """Object embeddings dataset for generating ZSL embeddings"""

    def __init__(self, cfg, split, mod):
        """
        Args:
            cfg: configs. Details can be found in
                zeroshoteval/config/defaults.py
            split(str): data split, e.g. `train`, `trainval`, `test`.
            mod(str): modality name
        """
        self.datadir = cfg.DATA.FEAT_EMB.PATH
        self.samples_per_class = cfg.ZSL.SAMPLES_PER_CLASS
        self.data = None
        self.labels = None

        assert split in ["trainval", "test", "test_unseen"]
        assert mod in ["CLSATTR", "IMG"]

        self.split = split
        self.mod = mod

        self._read_matdata()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample_label = self.labels[idx]
        sample_data = self.data[idx]

        return sample_data, sample_label

    def _read_matdata(self):
        """
        Reading mat dataset form root_dir
        """
        scaler = preprocessing.MinMaxScaler()
        att_splits = sio.loadmat("datasets/edgarschnfld_data/CUB/att_splits.mat")
        cnn_features = sio.loadmat("datasets/edgarschnfld_data/CUB/res101.mat")
        feature = cnn_features["features"].T
        labels = cnn_features["labels"].astype(int).squeeze() - 1

        # Load image modality data
        if self.mod in ["IMG"]:
            # Load trainval data
            if self.split in ["trainval"]:
                train_indices = att_splits["trainval_loc"].squeeze() - 1
                feature[train_indices] = scaler.fit_transform(
                    feature[train_indices]
                )
                indices_obj = np.array([], dtype=np.int64)
                seen_classes = np.unique(labels[train_indices])
                for label in seen_classes:
                    class_indices = np.intersect1d(
                        np.sort(train_indices),
                        np.where(labels == label),
                    )
                    class_indices = np.resize(
                        class_indices, self.samples_per_class.IMG
                    )
                    indices_obj = np.append(indices_obj, class_indices)

                self.data = feature[indices_obj]
                self.labels = labels[indices_obj]
            # Load test data
            elif self.split in ["test"]:
                test_seen_indices = att_splits["test_seen_loc"].squeeze() - 1
                test_unseen_indices = (
                        att_splits["test_unseen_loc"].squeeze() - 1
                )
                test_indices = np.append(test_seen_indices, test_unseen_indices)
                self.data = scaler.fit_transform(feature[test_indices])
                self.labels = labels[test_indices]
            # Load data for test unseen classes
            elif self.split in ["test_unseen"]:
                test_unseen_indices = (
                        att_splits["test_unseen_loc"].squeeze() - 1
                )

                self.data = scaler.fit_transform(feature[test_unseen_indices])
                self.labels = labels[test_unseen_indices]

        # Load class attributes modality data
        elif self.mod in ["CLSATTR"]:
            class_attr = att_splits["att"].T
            # Load data for test unseen classes
            if self.split in ["test_unseen"]:
                test_unseen_indices = (
                        att_splits["test_unseen_loc"].squeeze() - 1
                )
                unseen_classes = np.unique(labels[test_unseen_indices])
                usneen_classes_attr = class_attr[unseen_classes]
                self.data = np.resize(
                    usneen_classes_attr,
                    (
                        self.samples_per_class.CLSATTR * len(unseen_classes),
                        usneen_classes_attr.shape[1],
                    ),
                )
                self.labels = np.resize(
                    unseen_classes,
                    self.samples_per_class.CLSATTR * len(unseen_classes),
                )
