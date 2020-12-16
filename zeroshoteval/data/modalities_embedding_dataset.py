import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from scipy import io as sio
from sklearn import preprocessing
from torch import Tensor
from torch.utils.data import Dataset

from zeroshoteval.data.types import MODALITIES_TYPES

logger = logging.getLogger(__name__)


class ModalitiesEmbeddingDataset(Dataset):
    """
    Modalities Embedding Dataset.

    Contains precomputed embeddings of all images in specified dataset.

    TODO: fill in ModalitiesEmbeddingDataset docstrings
    """

    def __init__(self, root_dir: str, modalities: List[str], split: Optional[str] = None) -> None:
        """
        Args:
            root_dir: Directory that contains binary .npy files with precomputed embeddings.
            modalities: Modalities (keywords) to load from root directory (e.g. `IMG`, `CLSATTR`).
            split: Part of dataset to load. Possible values: `train`, `test`. If None returns the entire dataset. Splits
                   can be easily accessed with `splits` attribute.
        """
        assert os.path.isdir(root_dir), "Specified root directory must not be empty."
        assert set(modalities) <= set(MODALITIES_TYPES), f"Unknown modality! Available: {MODALITIES_TYPES}"
        assert split in ["train", "test"], f"Unknown data part! Available: {['train', 'test']}"

        self.root_dir = root_dir
        self.modalities = modalities

        self.splits: DataFrame = self._load_data_splits(split)
        self.data_indexes: np.ndarray = self.splits['id'].to_numpy()

        self.modalities_data: Dict[str, Tensor] = self._load_modalities_data()
        self.labels: np.ndarray = self._load_labels()

        self.seen_indexes: np.ndarray = self._parse_splits(is_seen=True)
        self.unseen_indexes: np.ndarray = self._parse_splits(is_seen=False)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, Tensor], Union[int, float]]:
        """
        Returns specified by index item from Dataset contents.

        Args:
            idx: Index of sample to return.

        Returns: Tuple2 with dictionary of `modalities names: item from each modality` and `sample label`

        """
        sample_label: Union[int, float] = self.labels[idx]
        sample_data: Dict[str, Tensor] = {}

        for modality_name, modality_data in self.modalities_data.items():
            sample_data[modality_name] = modality_data[idx]

        return sample_data, sample_label

    def _load_data_splits(self, split: str) -> DataFrame:
        root_dir = Path(self.root_dir)
        dataset_name: str = root_dir.parts[-1].split("_", 1)[0]
        splits_file: str = f"{dataset_name}_splits.csv"

        assert splits_file in os.listdir(self.root_dir), \
            f"Cannot find file with data splits! Make sure that file `{splits_file}` is presented in the root."

        splits_df: DataFrame = pd.read_csv(os.path.join(self.root_dir, splits_file))

        if split == "train":
            indexes: np.ndarray = (splits_df['is_train'] == 1).to_numpy()
        else:
            indexes: np.ndarray = (splits_df['is_train'] == 0).to_numpy()

        splits_df = splits_df.iloc[indexes]

        return splits_df

    def _parse_splits(self, is_seen: bool) -> np.ndarray:
        indexes: pd.Series = self.splits['is_seen'] == is_seen
        return self.splits['id'][indexes].to_numpy()

    def _load_modalities_data(self) -> Dict[str, Tensor]:
        """
        Loads modalities data from binary .npy files to NumPy arrays and stores it in RAM.
        After loading, all files are converted into Torch Tensors.

        Returns: Dictionary with `modalities keywords` as keys and `torch.tensor` as values

        """
        existing_modalities: List[str] = []
        file_paths: List[str] = []
        data: Dict[str, Tensor] = {}

        for file_name in os.listdir(self.root_dir):
            modality_name = file_name.split("_", 2)[1].upper()
            if modality_name in MODALITIES_TYPES:
                existing_modalities.append(modality_name)
                file_paths.append(os.path.join(self.root_dir, file_name))

        assert set(existing_modalities) >= set(self.modalities), \
            f"Some of specified modalities are not presented in data directory.\n" \
            f"Existing: {existing_modalities}\n" \
            f"Specified: {self.modalities}"

        for mod_name, mod_file in zip(existing_modalities, file_paths):
            # Load NumPy array from binary file
            mod_data_numpy: np.ndarray = np.load(mod_file)

            # Select only necessary data using indexes extracted from splits earlier
            mod_data_numpy = mod_data_numpy[self.data_indexes]

            # Convert to Torch Tensor
            mod_data: Tensor = torch.from_numpy(mod_data_numpy)
            data[mod_name] = mod_data

        return data

    def _load_labels(self) -> np.ndarray:
        root_dir = Path(self.root_dir)
        dataset_name: str = root_dir.parts[-1].split("_", 1)[0]
        labels_file: str = f"{dataset_name}_labels.npy"

        assert labels_file in os.listdir(self.root_dir), \
            f"Cannot find file with labels! Make sure that file `{labels_file}` is presented in the root."

        labels: np.ndarray = np.load(os.path.join(self.root_dir, labels_file))

        # Select only necessary data using indexes extracted from splits earlier
        labels = labels[self.data_indexes]

        return labels


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
        Reading mat dataset form root_dir
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
