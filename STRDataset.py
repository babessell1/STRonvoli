import os
import numpy as np
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
        locus = self.metadata['trid'].iloc[idx]
        sample_name = self.metadata['sample_name'].iloc[idx]

        ohe_file = os.path.join(self.ohe_dir, sample_name, f'{sample_name}_{locus}.npy')
        ohe = np.load(ohe_file)
        
        mc = self.metadata['MC'].iloc[idx]
        mc_split = mc.str.split(',', expand=True).astype(float)
        label = mc_split.apply(lambda row: max(row[0], row[1]), axis=1)
        
        metadata = self.metadata.iloc[idx]

        return ohe, label, metadata