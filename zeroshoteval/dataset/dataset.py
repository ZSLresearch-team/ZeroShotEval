import numpy as np
import torch
from scipy import io as sio
from sklearn import preprocessing
from torch.utils.data import Dataset

import logging

logger = logging.getLogger(__name__)


class ObjEmbeddingDataset(Dataset):
    """Object embeddings dataset"""

    def __init__(self, cfg, object_modalities, split):
        """
        Args:
            cfg(CfgNode): configs. Detail can de found in
            zeroshoteval/config/defaults.py
            object_modalities: list of modalities names, wich describes
                objects, not classes.
            split(str): data split, e.g. `train`, `trainval`, `test`.
        """
        self.datadir = cfg.DATA.FEAT_EMB.PATH
        self.data = None
        self.labels = None

        assert split in ["trainval", "test"]
        self.split = split

        self.train_indices = None
        self.test_indices = None
        self.test_seen_indices = None
        self.test_unseen_indices = None

        self._read_matdata()

        self.seen_classes = np.unique(self.la[self.train_indices])
        self.unseen_classes = np.unique(self.la[self.test_unseen_indices])
        self.num_seen_classes = len(self.seen_classes)
        self.num_unseen_classes = len(self.unseen_classes)
        self.num_classes = len(self.seen_classes) + len(self.unseen_classes)

        self.seen_class_mapping = {
            old_label: new_label
            for old_label, new_label in zip(
                self.seen_classes, np.arange(self.num_seen_classes)
            )
        }
        self.unseen_class_mapping = {
            old_label: new_label
            for old_label, new_label in zip(
                self.unseen_classes, np.arange(self.num_unseen_classes)
            )
        }
        self.modalities = [mod.upper() for mod in self.data.keys()]
        self.object_modalities = object_modalities
        self.class_modalities = list(
            set(self.modalities) - set(object_modalities)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample_label = self.labels[idx]
        sample_data = {}

        for modality_name, modality_data in self.data.items():
            if modality_name in self.object_modalities:
                sample_data[modality_name] = modality_data[idx]

            elif modality_name in self.class_modalities:
                sample_data[modality_name] = modality_data[sample_label]

            else:
                raise Exception(f"No {modality_name} in modalities list")

        return sample_data, sample_label

    def _read_matdata(self):
        """
        Loads data from mat files
        """
        # Getting CNN features and labels
        logger.info(
            f"Loading computed {self.split} split embeddings "
            f"from {self.datadir}... "
        )

        cnn_features = sio.loadmat(self.datadir + "res101.mat")
        feature = cnn_features["features"].T
        labels = cnn_features["labels"].astype(int).squeeze() - 1

        # Getting data splits and class attributes
        att_splits = sio.loadmat(self.datadir + "att_splits.mat")
        self.train_indices = att_splits["trainval_loc"].squeeze() - 1
        self.test_seen_indices = att_splits["test_seen_loc"].squeeze() - 1
        self.test_unseen_indices = att_splits["test_unseen_loc"].squeeze() - 1
        self.test_indices = None
        self.la = labels
        # numpy array index starts from 0, matlab starts from 1
        if self.split in ["trainval"]:
            indices = (
                att_splits["trainval_loc"].squeeze() - 1
            )  # --> train_feature = TRAIN SEEN
        elif self.split in ["test"]:
            test_seen_indices = att_splits["test_seen_loc"].squeeze() - 1
            test_unseen_indices = att_splits["test_unseen_loc"].squeeze() - 1
            indices = np.append(test_seen_indices, test_unseen_indices)

        class_attr = att_splits["att"].T

        scaler = preprocessing.MinMaxScaler()
        feature = scaler.fit_transform(feature[indices])
        self.labels = labels[indices]

        data = {}
        data["IMG"] = feature
        data["CLS_ATTR"] = class_attr

        self.data = data

        logger.info("Embeddings succesfully loaded.")

        return None


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
        assert mod in ["CLS_ATTR", "IMG"]

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
        Reading mat dataset form datadir
        """
        scaler = preprocessing.MinMaxScaler()
        att_splits = sio.loadmat(self.datadir + "att_splits.mat")
        cnn_features = sio.loadmat(self.datadir + "res101.mat")
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
                        np.sort(train_indices), np.where(labels == label),
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
        elif self.mod in ["CLS_ATTR"]:
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
                        self.samples_per_class.CLS_ATTR * len(unseen_classes),
                        usneen_classes_attr.shape[1],
                    ),
                )
                self.labels = np.resize(
                    unseen_classes,
                    self.samples_per_class.CLS_ATTR * len(unseen_classes),
                )
