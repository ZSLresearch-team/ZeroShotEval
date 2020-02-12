"""The main script for loading data into ZeroShotEval training procedure.

The script imports all loaders for all possible datasets
and recieves at the input name of dataset and modalities to load.
Then based on the input parameters loads necessary data and
returns it.
"""

# from cub_loader import load_cub
# from awa2_loader import load_awa2
# ...
# from cub_awa_sun_emb_loader import ...


def load_dataset(dataset_name, modalities, path, load_embeddings=False):
    if dataset_name == 'cub':
        if load_embeddings:
            # return load embeddings for cub
            pass
        # return load_cub(modalities, path)
        pass
    
    elif dataset_name == 'awa2':
        if load_embeddings:
            # return load embeddings for awa2
            pass
        # return load_awa2(modalities, path)
        pass
    
    # elif dataset_name ==
    # ...
