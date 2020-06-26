"""Script for raw data loading from the dataset AWA2 (Animals With Attributes 2).
Link: https://cvml.ist.ac.at/AwA2/

The script performs 3 tasks:
1. For a separate data (e.g. images) creates a pandas.DataFrame 'img_paths_df'
with relative paths (from root) to each file in the dataset;
2. For a data packed in one file (e.g. attributes) reads this file
and stores data in the memory;
3. Extracts dataset splits into train/test and seen/unseen parts and
creates a pandas.DataFrame 'splits_df' with image name and 4 columns
(train | test | seen | unseen) with 0/1 (False/True) for each image.

NOTE: the script should recieve at the input names of modalities to load
e.g. modalities=['images', 'attributes']
"""


def load_awa2(modalities, path):
    pass
