import torch
from torch.utils.data import Dataset

class ObjEmbeddingDataset(Dataset):
    """Object emd dataset"""

    def __init__(self, data, labels):
        """
        Args:
            data(dict): dict mapping modalities name to modalities object embeddings.
            labels: list of ground truth labels for objects.
        """
        self.data = data
        data = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_label = self.labels[idx]
        sample_data = {}

        for modality in self.data.keys():
            sample_data[modality] = self.data[modality][idx]

        return sample_data, sample_label
