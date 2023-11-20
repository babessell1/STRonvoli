import os
import pandas as pd
from torch.utils import data

class STRDataset(data.Dataset):
    """
    Custom PyTorch dataset class mapping short-read STR loci and their corresponding features to long-read STR counts
    """
    def __init__(self, ohe_dir, metadata_file):
        self.ohe_dir = ohe_dir # each file is a one-hot encoding of the sequence and depth at a locus
        self.metadata = pd.read_csv(metadata_file) # contains truth labels and metadata

    def __len__(self):
        return len(self.annots)
    
    def __getitem__(self, idx):
        locus = self.metadata['locus'].iloc[idx]
        ohe = os.path.join(self.ohe_dir, f'{locus}.npy')
        label = self.metadata['labels'].iloc[idx]
        metadata = self.metadata.iloc[idx, 2:]

        return ohe, label, metadata