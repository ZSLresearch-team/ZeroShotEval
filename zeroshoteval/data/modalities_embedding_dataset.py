import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
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

    scaler = None

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
        self.modalities = modalities  # TODO: param does not do anything!

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
        splits: DataFrame = self.splits.reset_index()
        indexes = splits.index[splits['is_seen'] == is_seen]
        return indexes.to_numpy()

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
                
                # TODO: are modalities type fixed for all datasets? What if specified modality is not presented in dataset and not necessary?
                
                existing_modalities.append(modality_name)
                file_paths.append(os.path.join(self.root_dir, file_name))

        assert set(existing_modalities) >= set(self.modalities), \
            f"Some of specified modalities are not presented in data directory.\n" \
            f"Existing: {existing_modalities}\n" \
            f"Specified: {self.modalities}"

        for mod_name, mod_file in zip(existing_modalities, file_paths):
            # Load NumPy array from binary file
            mod_data_numpy: np.ndarray = np.load(mod_file)

            # ----------------------------------

            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()

            mod_data_numpy = scaler.fit_transform(mod_data_numpy)

            # ----------------------------------

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
