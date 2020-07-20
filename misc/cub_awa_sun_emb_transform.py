"""Script for precomputed embeddings loading for 
the datasets: CUB, SUN, AWA1, AWA2.

The script loads precomputed embeddings from .mat file and 
transforms it into a unified format (pickle).

NOTE: Embeddings were computed using ResNet101 network pretrained on ImageNet
The data provided by unofficial resourse and packed into a single .mat file.

!!! IMPORTANT !!!
Please note, that this script is designed for CUB, SUN, AWA1, AWA2 
datasets only, and moreover these datasets must be downloaded from: 
https://www.dropbox.com/sh/btoc495ytfbnbat/AAAaurkoKnnk0uV-swgF-gdSa?dl=0

In other case you will catch a lot of errors!
"""

import numpy as np
import pandas as pd
from scipy import io as sio
from sklearn import preprocessing

# region IMPORTS
import argparse
import json
import os
import pickle
from pathlib import Path

# endregion


def init_arguments():
    parser = argparse.ArgumentParser(
        description="Script for embeddings transformation to pickle file"
    )
    parser.add_argument(
        "--dataset", required=True, help="Name of the dataset to transform."
    )
    parser.add_argument(
        "--path", required=True, help="Path to the dataset to transform."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to the output directory where the transformed \
                            embeddings will be saved.",
    )
    return parser


def load_arguments():
    parser = init_arguments()
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.path = Path(args.path)
    args.output_dir = Path(args.output_dir)

    return args


def read_matdataset(matdata_file, root_path):
    print("\nLoading image data from .mat files...", end=" ")
    matdata = sio.loadmat(os.path.join(root_path, matdata_file))

    img_embeddings = matdata["features"].T
    img_embeddings = preprocess_data(img_embeddings)
    labels = matdata["labels"].astype(int).squeeze() - 1

    img_embeddings = {"img": img_embeddings}
    print("Done!")
    return img_embeddings, labels


def read_matattributes(matattrsplit_file, root_path, dataset):
    print("\nLoading attributes from .mat files...", end=" ")
    aux_modalities_embs = {}
    matattributes = sio.loadmat(os.path.join(root_path, matattrsplit_file))

    aux_modalities_embs["cls_attr"] = matattributes["att"].T

    if dataset == "cub":
        with open(root_path / "CUB_supporting_data.p", "rb") as h:
            x = pickle.load(h)
            for key, value in x.items():
                aux_modalities_embs[key] = value
    print("Done!")
    return aux_modalities_embs


def read_data_splits(matattrsplit_file, root_path):
    print("\nLoading data splits from .mat files...", end=" ")
    splits_df = pd.DataFrame(columns=["obj_id", "is_train", "is_seen"])

    matattrsplit = sio.loadmat(os.path.join(root_path, matattrsplit_file))

    # NOTE: numpy array index starts from 0, matlab starts from 1
    trainval_loc = matattrsplit["trainval_loc"].squeeze() - 1
    test_seen_loc = matattrsplit["test_seen_loc"].squeeze() - 1
    test_unseen_loc = matattrsplit["test_unseen_loc"].squeeze() - 1

    # NOTE: there are two more fileds in .mat files that
    # are not in use now, but can be used
    # train_loc = matattrsplit['train_loc'].squeeze() - 1  #--> train_feature = TRAIN SEEN
    # val_unseen_loc = matattrsplit['val_loc'].squeeze() - 1  #--> test_unseen_feature = TEST UNSEEN

    for part, params in zip(
        (trainval_loc, test_seen_loc, test_unseen_loc), ((1, 1), (0, 1), (0, 0))
    ):
        part_df = pd.DataFrame(columns=splits_df.columns)
        part_df.loc[:, "obj_id"] = np.sort(part)
        part_df.loc[:, "is_train"] = params[0]
        part_df.loc[:, "is_seen"] = params[1]
        splits_df = splits_df.append(part_df)

    print("Done!")
    return splits_df


def preprocess_data(data):
    print("Preprocessing data using MinMaxScaler...", end=" ")
    scaler = preprocessing.MinMaxScaler()
    print("Done!")
    return scaler.fit_transform(data)


def load_dataset_embeddings(
    dataset_name,
    path,
    matdata_file="res101.mat",
    matattrsplit_file="att_splits.mat",
):
    """Loads specified dataset to dictionary structure 
    for further saving to pickle file.
    """
    valid_datasets = ["cub", "sun", "awa1", "awa2"]
    if dataset_name not in valid_datasets:
        raise ValueError(
            "Unknown dataset! This script is aimed on {} datasets".format(
                valid_datasets
            )
        )

    print("\nLoading dataset {} from {}".format(dataset_name, path))

    print(
        "\n[INFO]  Please note, that this script is designed for \
            \n\tCUB, SUN, AWA1, AWA2 datasets only, and moreover \
            \n\tthese datasets must be downloaded from: \
            \n\thttps://www.dropbox.com/sh/btoc495ytfbnbat/AAAaurkoKnnk0uV-swgF-gdSa?dl=0 \
            \n\tIn other case you will catch a lot of errors!"
    )

    img_embeddings, labels = read_matdataset(matdata_file, root_path=path)

    aux_modalities_embeddings = read_matattributes(
        matattrsplit_file, root_path=path, dataset=dataset_name
    )
    splits_df = read_data_splits(matattrsplit_file, root_path=path)

    embeddings_dict = {**img_embeddings, **aux_modalities_embeddings}

    return embeddings_dict, labels, splits_df


def make_directory(path):
    directory = Path(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)


def save_data(embeddings, labels, splits_df, dataset_name, save_dir):
    make_directory(save_dir)
    embs_path = save_dir / (dataset_name + "_obj_embeddings.pickle")
    lab_path = save_dir / (dataset_name + "_obj_labels.pickle")
    splits_path = save_dir / (dataset_name + "_splits.csv")

    print('\nSaving embeddings to "{}"'.format(embs_path))
    with open(embs_path, "wb") as f:
        pickle.dump(embeddings, f)

    print('Saving labels to "{}"'.format(lab_path))
    with open(lab_path, "wb") as f:
        pickle.dump(labels, f)

    print('Save data splits to "{}"'.format(splits_path))
    splits_df.to_csv(splits_path, index=False)
    print("Done!")


if __name__ == "__main__":
    args = load_arguments()
    embeddings, labels, splits_df = load_dataset_embeddings(
        dataset_name=args.dataset, path=args.path
    )

    save_data(
        embeddings,
        labels,
        splits_df,
        dataset_name=args.dataset,
        save_dir=args.output_dir,
    )
