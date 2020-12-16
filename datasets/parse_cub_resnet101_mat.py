"""Script for precomputed embeddings loading for 
the datasets: CUB, SUN, AWA1, AWA2.

The script loads precomputed embeddings from .mat file and 
transforms it into a unified format (pickle).

NOTE: Embeddings were computed using ResNet101 network pretrained on ImageNet
The data provided by unofficial resource and packed into a single .mat file.

!!! IMPORTANT !!!
Please note, that this script is designed for CUB, SUN, AWA1, AWA2 
datasets only, and moreover these datasets must be downloaded from: 
https://www.dropbox.com/sh/btoc495ytfbnbat/AAAaurkoKnnk0uV-swgF-gdSa?dl=0
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import io as sio

from zeroshoteval.utils import setup_logger

logger = setup_logger()


def init_arguments():
    parser = argparse.ArgumentParser(description="Script for embeddings transformation to pickle file")
    parser.add_argument("--path", required=True,
                        help="Path to the dataset to transform.")
    parser.add_argument("--output-dir", required=True,
                        help="Path to the output directory where the transformed embeddings will be saved.")
    return parser


def load_arguments():
    parser = init_arguments()
    args = parser.parse_args()

    args.path = Path(args.path)
    args.output_dir = Path(args.output_dir)

    return args


def read_data(images_mat_file: str,
              cls_attributes_mat_file: str,
              root_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat_data: Dict = sio.loadmat(os.path.join(root_path, images_mat_file))
    img_data: np.ndarray = mat_data["features"].T

    # NOTE: numpy array index starts from 0, matlab starts from 1
    labels_data: np.ndarray = mat_data["labels"].astype(int).squeeze() - 1

    mat_data: Dict = sio.loadmat(os.path.join(root_path, cls_attributes_mat_file))
    cls_attr_data: np.ndarray = mat_data["att"].T

    cls_attr_data = extend_clsattr_data(cls_attr_data, labels_data)

    # TODO: deal with supporting data
    # if dataset == "CUB":
    #     with open(root_path / "CUB_supporting_data.p", "rb") as h:
    #         x = pickle.load(h)
    #         for key, value in x.items():
    #             aux_modalities_embs[key] = value

    logger.info("Image data, class attributes data and labels were successfully parsed.")
    return img_data, cls_attr_data, labels_data


def extend_clsattr_data(cls_attr_data: np.ndarray, labels_data: np.ndarray) -> np.ndarray:
    """
    Extends clsattr_data from shape (num_classes, emb_size) to (num_instances, emb_size) by duplicating class embeddings
    according to labels in labels_data.

    Args:
        cls_attr_data: Array with embeddings of class attributes.
        labels_data: Array with dataset labels.

    Returns: Array with extended clsattr_data.

    """
    if len(cls_attr_data) == len(labels_data):
        logger.warning("Class attributes data seems to be already extended. Length of array is the same as labels.")
        return cls_attr_data
    else:
        extended_clsattr = [cls_attr_data[label] for label in labels_data]
        logger.debug("Class attributes data were successfully extended according to labels data.")
        return np.stack(extended_clsattr)


def read_data_splits(splits_mat_file: str, root_path: str) -> pd.DataFrame:
    mat_data = sio.loadmat(os.path.join(root_path, splits_mat_file))

    df = pd.DataFrame(columns=["id", "is_train", "is_seen"])

    # NOTE: numpy array index starts from 0, matlab starts from 1
    train_indexes = mat_data["trainval_loc"].squeeze() - 1
    test_seen_indexes = mat_data["test_seen_loc"].squeeze() - 1
    test_unseen_indexes = mat_data["test_unseen_loc"].squeeze() - 1

    # NOTE: there are two more fields in mat files that are not in use now, but can be used
    # train_loc = mat_data['train_loc'].squeeze() - 1  #--> train_feature = TRAIN SEEN
    # val_unseen_loc = mat_data['val_loc'].squeeze() - 1  #--> test_unseen_feature = TEST UNSEEN

    for part, params in zip((train_indexes, test_seen_indexes, test_unseen_indexes), ((1, 1), (0, 1), (0, 0))):
        part_df = pd.DataFrame(columns=df.columns)
        part_df.loc[:, "id"] = part
        part_df.loc[:, "is_train"] = params[0]
        part_df.loc[:, "is_seen"] = params[1]
        df = df.append(part_df)

    df.reset_index(drop=True, inplace=True)

    logger.info("Data splits were successfully parsed.")
    return df


def parse_mat_dataset(path: str, mat_data_file="res101.mat", mat_attributes_file="att_splits.mat"):
    """
    Loads specified dataset to dictionary structure
    for further saving to pickle file.
    """
    path = Path(path)
    dataset: str = path.name.upper()
    valid_datasets = ["CUB", "SUN", "AWA1", "AWA2"]
    assert dataset in valid_datasets, \
        f"Unknown dataset! This script is aimed on datasets: {valid_datasets}"

    logger.info(f"Parsing dataset {dataset} from mat files into NumPy arrays. Source path: {path}")

    img_data, cls_attr_data, labels_data = read_data(mat_data_file, mat_attributes_file, root_path=path)

    splits_df = read_data_splits(mat_attributes_file, root_path=path)

    return img_data, cls_attr_data, labels_data, splits_df


def save_data(img_data: np.ndarray,
              cls_attr_data: np.ndarray,
              labels_data: np.ndarray,
              splits_df: pd.DataFrame,
              dataset: str,
              save_dir: str):
    save_dir = Path(save_dir)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    np.save(file=save_dir / f"{dataset}_img_embeddings", arr=img_data)
    np.save(file=save_dir / f"{dataset}_clsattr_embeddings", arr=cls_attr_data)
    np.save(file=save_dir / f"{dataset}_labels", arr=labels_data)

    splits_df.to_csv(save_dir / f"{dataset}_splits.csv", index=False)

    logger.info(f"Parsed data for {dataset} dataset was successfully saved to {save_dir.absolute()}")


if __name__ == "__main__":
    args = load_arguments()

    img, cls_attr, labels, splits = parse_mat_dataset(path=args.path)

    dataset_name = Path(args.path).name
    save_data(img, cls_attr, labels, splits, dataset_name, save_dir=args.output_dir)
