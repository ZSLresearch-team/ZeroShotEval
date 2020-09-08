import numpy as np
import torch
from scipy import io as sio
from sklearn import preprocessing
from torch.utils.data import Dataset

import logging

logger = logging.getLogger(__name__)


class ObjEmbeddingDataset(Dataset):
    """Object embeddings dataset"""

    def __init__(self, datadir, object_modalities, split):
        """
        Args:
            data(dict): dict mapping modalities name to modalities object
                embeddings.
            labels: list of ground truth labels for objects.
            split_df: pandas dataframe with dataset splits
            object_modalities: list of modalities names, wich describes
                objects, not classes.
        """
        self.datadir = datadir
        self.data = None
        self.labels = None
        
        assert split in ["trainval", "test", "ce"]
        self.split = split

        self.train_indices = None
        self.test_indices = None
        self.test_seen_indices = None
        self.test_unseen_indices = None

        self.read_matdata()

        # self.seen_classes = np.unique(self.labels[self.train_indices])
        # self.unseen_classes = np.unique(self.labels[self.test_unseen_indices])
        # self.num_seen_classes = len(self.seen_classes)
        # self.num_unseen_classes = len(self.unseen_classes)
        # self.num_classes = len(self.seen_classes) + len(self.unseen_classes)

        # self.seen_class_mapping = {
        #     old_label: new_label
        #     for old_label, new_label in zip(
        #         self.seen_classes, np.arange(self.num_seen_classes)
        #     )
        # }
        # self.unseen_class_mapping = {
        #     old_label: new_label
        #     for old_label, new_label in zip(
        #         self.unseen_classes, np.arange(self.num_unseen_classes)
        #     )
        # }
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


    def get_zsl_emb_indice(self, samples_per_modality_class, generalized=True):
        r"""
        Creates indices of dataset to create zsl embeddings dataset.

        Args:
            samples_per_modality_class(CfgNode): samples generated per class
                for each modality.
            generalized(bool): if ``True`` get indices for generalized mod,
                if ``False`` -
            for few-shot learning

        Returns:
            indices_obj: array with indices of object modality
            indices_class: array with indices of class modality
        """
        # !replase with dict : modality_name -> indices
        indices_obj = np.array([], dtype=np.int64)
        usneen_classes_emb = np.array([], dtype=np.float64)
        usneen_classes_emb_labels = np.array([], dtype=np.int64)

        for (
            modality_name,
            samples_per_class,
        ) in samples_per_modality_class.items():
            if (modality_name in self.object_modalities) and generalized:
                for label in self.seen_classes:
                    class_indices = np.intersect1d(
                        np.sort(self.train_indices),
                        np.where(self.labels == label),
                    )

                    class_indices = np.resize(class_indices, samples_per_class)
                    indices_obj = np.append(indices_obj, class_indices)

            else:

                usneen_classes_emb = self.data[modality_name][
                    self.unseen_classes
                ]
                if not generalized:
                    unseen_labels = np.arange(0, len(self.unseen_classes))
                else:
                    unseen_labels = self.unseen_classes

                usneen_classes_emb_labels = np.append(
                    usneen_classes_emb_labels, unseen_labels
                )

                usneen_classes_emb = np.resize(
                    usneen_classes_emb,
                    (
                        samples_per_class * self.num_unseen_classes,
                        usneen_classes_emb.shape[1],
                    ),
                )
                usneen_classes_emb_labels = np.resize(
                    usneen_classes_emb_labels,
                    samples_per_class * self.num_unseen_classes,
                )

        return indices_obj, usneen_classes_emb, usneen_classes_emb_labels

    def read_matdata(self):
        """
        Reading mat dataset form datadir
        """
        # Getting CNN features and labels
        logger.info(
            f"Loading computed embeddings from {self.datadir}... "
            f"with {self.split} split"
        )

        cnn_features = sio.loadmat(self.datadir + "res101.mat")
        feature = cnn_features["features"].T
        labels = cnn_features["labels"].astype(int).squeeze() - 1

        # Getting data splits and class attributes
        att_splits = sio.loadmat(self.datadir + "att_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        if self.split in ["trainval"]:
            indices = (
                att_splits["trainval_loc"].squeeze() - 1
            )  # --> train_feature = TRAIN SEEN
        elif self.split in ["test"]:
            test_seen_indices = att_splits["test_seen_loc"].squeeze() - 1
            test_unseen_indices = att_splits["test_unseen_loc"].squeeze() - 1
            indices = np.append(
                test_seen_indices, test_unseen_indices
            )
    
        class_attr = torch.from_numpy(att_splits["att"].T)

        scaler = preprocessing.MinMaxScaler()
        if self.split =="ce":
            feature = scaler.fit_transform(feature)
            self.labels = labels
        else:
            feature = scaler.fit_transform(feature[indices])
            self.labels = labels[indices]

        data = {}
        data["IMG"] = feature
        data["CLS_ATTR"] = class_attr

        self.data = data

        logger.info("Embeddings succesfully loaded.")

        return None


class GenEmbeddingDataset(Dataset):
    """Object embeddings dataset"""

    def __init__(self, cfg, split):
        """
        Args:
            data(dict): dict mapping modalities name to modalities object
                embeddings.
            labels: list of ground truth labels for objects.
            split_df: pandas dataframe with dataset splits
            object_modalities: list of modalities names, wich describes
                objects, not classes.
        """
        self.datadir = cfg.DATA.FEAT_EMB.PATH
        self.split = split
        self.data = None
        self.labels = None

        assert split in ["trainval", "test", "ce"]
        self.split = split

        self.train_indices = None
        self.test_indices = None
        self.test_seen_indices = None
        self.test_unseen_indices = None   

        self.read_matdata()


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):

        sample_label = self.labels[idx]
        sample_data = self.data[idx]

        return sample_data, sample_label


    def get_zsl_emb_indice(self, samples_per_modality_class, generalized=True):
        r"""
        Creates indices of dataset to create zsl embeddings dataset.

        Args:
            samples_per_modality_class(CfgNode): samples generated per class
                for each modality.
            generalized(bool): if ``True`` get indices for generalized mod,
                if ``False`` -
            for few-shot learning

        Returns:
            indices_obj: array with indices of object modality
            indices_class: array with indices of class modality
        """
        # !replase with dict : modality_name -> indices
        indices_obj = np.array([], dtype=np.int64)
        usneen_classes_emb = np.array([], dtype=np.float64)
        usneen_classes_emb_labels = np.array([], dtype=np.int64)

        for (
            modality_name,
            samples_per_class,
        ) in samples_per_modality_class.items():
            if (modality_name in self.object_modalities) and generalized:
                for label in self.seen_classes:
                    class_indices = np.intersect1d(
                        np.sort(self.train_indices),
                        np.where(self.labels == label),
                    )

                    class_indices = np.resize(class_indices, samples_per_class)
                    indices_obj = np.append(indices_obj, class_indices)

            else:

                usneen_classes_emb = self.data[modality_name][
                    self.unseen_classes
                ]
                if not generalized:
                    unseen_labels = np.arange(0, len(self.unseen_classes))
                else:
                    unseen_labels = self.unseen_classes

                usneen_classes_emb_labels = np.append(
                    usneen_classes_emb_labels, unseen_labels
                )

                usneen_classes_emb = np.resize(
                    usneen_classes_emb,
                    (
                        samples_per_class * self.num_unseen_classes,
                        usneen_classes_emb.shape[1],
                    ),
                )
                usneen_classes_emb_labels = np.resize(
                    usneen_classes_emb_labels,
                    samples_per_class * self.num_unseen_classes,
                )

        return indices_obj, usneen_classes_emb, usneen_classes_emb_labels


    def read_matdata(self):
        """
        Reading mat dataset form datadir
        """
        # Getting CNN features and labels
        logger.info(
            f"Loading computed embeddings from {self.datadir}... "
            f"with {self.split} split"
        )
        att_splits = sio.loadmat(self.datadir + "att_splits.mat")
        if self.split in ['trainval']:
            train_indices = (
                att_splits["trainval_loc"].squeeze() - 1
            ) 
            indices_obj = np.array([], dtype=np.int64)

            cnn_features = sio.loadmat(self.datadir + "res101.mat")
            feature = cnn_features["features"].T
            labels = cnn_features["labels"].astype(int).squeeze() - 1
            seen_classes = np.unique(labels[train_indices])
            for label in seen_classes: 
                class_indices = np.intersect1d(
                    np.sort(train_indices),
                    np.where(labels == label),
                )
            class_indices = np.resize(class_indices, 50)
            indices_obj = np.append(indices_obj, class_indices)
            self.data = feature[indices_obj,]
            self.labels = labels[indices_obj,]

        logger.info("Embeddings succesfully loaded.")

        return None
