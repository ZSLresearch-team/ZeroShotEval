import numpy as np
import scipy.io as sio
from sklearn import preprocessing

import torch
from torch.utils.data import Dataset


class ObjEmbeddingDataset(Dataset):
    """Object embeddings dataset"""

    def __init__(self, datadir, object_modalities, verbose=1):
        """
        Args:
            data(dict): dict mapping modalities name to modalities object embeddings.
            labels: list of ground truth labels for objects.
            split_df: pandas dataframe with dataset splits
            object_modalities: list of modalities names, wich describes objects, not classes.
        """
        self.datadir = datadir
        self.data = None
        self.labels = None


        self.train_indices = None
        self.test_indices = None
        self.test_seen_indices = None
        self.test_unseen_indices = None

        self.read_matdata(verbose)

        self.seen_classes = np.unique(self.labels[self.train_indices])
        self.unseen_classes = np.unique(self.labels[self.test_unseen_indices])
        self.num_seen_classes = len(self.seen_classes)
        self.num_unseen_classes = len(self.unseen_classes)
        self.num_classes = len(self.seen_classes) + len(self.unseen_classes)

        self.seen_class_mapping = {old_label:new_label for old_label, new_label in zip(self.seen_classes,
                                                                                       np.arange(self.num_seen_classes))}
        self.unseen_class_mapping = {old_label:new_label for old_label, new_label in zip(self.unseen_classes,
                                                                                       np.arange(self.num_unseen_classes))}
        self.modalities = self.data.keys()
        self.object_modalities = object_modalities
        self.class_modalities = list(set(self.modalities) - set(object_modalities))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

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
        # !replase with dict : modality_name -> indices
        indices_obj = np.array([], dtype=np.int64)
        usneen_classes_emb = np.array([], dtype=np.float64)
        usneen_classes_emb_labels = np.array([], dtype=np.int64)

        for modality_name, samples_per_class in samples_per_modality_class.items():
            if (modality_name in self.object_modalities) and generalized:
                for label in self.seen_classes:
                    class_indices = np.intersect1d(np.sort(self.train_indices), np.where(self.labels == label))

                    class_indices = np.resize(class_indices, samples_per_class)
                    indices_obj = np.append(indices_obj, class_indices)

            else:

                usneen_classes_emb = self.data[modality_name][self.unseen_classes]
                if not generalized:
                    unseen_labels = np.arange(0, len(self.unseen_classes))
                else:
                    unseen_labels = self.unseen_classes

                usneen_classes_emb_labels = np.append(usneen_classes_emb_labels, unseen_labels)

                usneen_classes_emb = np.resize(usneen_classes_emb,
                                               (samples_per_class * self.num_unseen_classes, usneen_classes_emb.shape[1]))
                usneen_classes_emb_labels = np.resize(usneen_classes_emb_labels, samples_per_class * self.num_unseen_classes)


        return indices_obj, usneen_classes_emb, usneen_classes_emb_labels



    def read_matdata(self, verbose=1):
        """
        Reading mat dataset form datadir
        """
        # Getting CNN features and labels
        if verbose > 1:
            print(f"[INFO] Loading computed embeddings and splits from {self.datadir}...")

        cnn_features = sio.loadmat(self.datadir + 'resnet101/res101.mat')
        feature = cnn_features['features'].T
        self.labels = cnn_features['labels'].astype(int).squeeze() - 1

        # Getting data splits and class attributes
        att_splits = sio.loadmat(self.datadir + 'resnet101/att_splits.mat')
        # numpy array index starts from 0, matlab starts from 1
        self.train_indices = att_splits['trainval_loc'].squeeze() - 1 #--> train_feature = TRAIN SEEN
        self.test_seen_indices = att_splits['test_seen_loc'].squeeze() - 1
        self.test_unseen_indices = att_splits['test_unseen_loc'].squeeze() - 1
        self.test_indices = np.append(self.test_seen_indices, self.test_unseen_indices)

        class_attr = torch.from_numpy(att_splits['att'].T)

        scaler = preprocessing.MinMaxScaler()

        feature = scaler.fit_transform(feature)

        data = {}
        data['img'] = feature
        data['cls_attr'] = class_attr

        self.data = data

        if verbose > 1:
            print('Done!')

        return None
