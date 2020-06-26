import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_zsl_emb(datadir):
    """
    """
    train_X = torch.load(datadir + "zsl_emb/train_X.pt")
    train_Y = torch.load(datadir + "zsl_emb/train_Y.pt")
    test_seen_X = torch.load(datadir + "zsl_emb/test_seen_X.pt")
    test_seen_Y = torch.load(datadir + "zsl_emb/test_seen_Y.pt")
    test_novel_X = torch.load(datadir + "zsl_emb/test_novel_X.pt")
    test_novel_Y = torch.load(datadir + "zsl_emb/test_novel_Y.pt")

    zsl_emb = torch.cat((train_X, test_seen_X, test_novel_X), 0)

    labels_tensor = torch.cat((train_Y, test_seen_Y, test_novel_Y), 0)

    # Getting train and test indices
    n_train = len(train_Y)
    csl_train_indice = np.arange(n_train)
    csl_test_indice = np.arange(
        n_train, n_train + len(test_seen_Y) + len(test_novel_Y)
    )

    zsl_emb_dataset = TensorDataset(zsl_emb, labels_tensor)

    return zsl_emb_dataset, csl_train_indice, csl_test_indice
