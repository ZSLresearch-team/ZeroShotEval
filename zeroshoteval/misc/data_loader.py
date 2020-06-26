"""The main script for loading data into ZeroShotEval training procedure.

The script imports all loaders for all possible datasets
and recieves at the input name of dataset and modalities to load.
Then based on the input parameters loads necessary data and
returns it.
"""

# from cub_loader import load_cub
# from awa2_loader import load_awa2
# ...


def load_dataset(dataset_name, modalities, path):
    if dataset_name == "cub":
        # return load_cub(modalities, path)
        pass

    elif dataset_name == "awa2":
        # return load_awa2(modalities, path)
        pass

    # NOTE: add other datasets loading here
    # elif dataset_name ==
    # ...
