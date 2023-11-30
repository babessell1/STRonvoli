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

        flanking_reads_vectorized = self.vectorize_flanking(metadata) # 50
        inrepeat_reads_sum = self.sum_inrepeat(metadata) # 1
        spanning_reads_sum = self.sum_spanning(metadata) # 1
        motif_vec = self.vectorize_motif(metadata) # 24
        motif_len = metadata['len_motif'] # 1
        repeat_region_len = metadata['len_repeat_region'] # 1

        mlp_array = np.array([flanking_reads_vectorized, inrepeat_reads_sum, spanning_reads_sum, motif_vec, motif_len, repeat_region_len]) # len(mlp_array) == 78

        return ohe, label, mlp_array
    
    def vectorize_motif(metadata):
        base_dict = {
            'A': 1,
            'C': 2,
            'T': 3,
            'G': 4
        }
        motif = metadata['motif']
        motif_arr = np.zeros(24) # longest motif in our dataset
        for i, char in enumerate(motif):
            motif_arr[i] = base_dict[char]
        return motif_arr
    
    def vectorize_flanking(metadata):
        flanking_reads = metadata['flanking_reads'].replace(' ', '').split('),(')
        flanking_reads_list = [tuple(map(int, t.strip('()').split(','))) for t in flanking_reads if t!= '']
        filtered_flanking_reads = [t for t in flanking_reads_list if t[0] <= 50] # number of loci with flanking reads in >50 repeats drastically drops off
        flanking_reads_vectorized = np.zeros(50)
        for index, value in filtered_flanking_reads:
            flanking_reads_vectorized[index - 1] = value
        return flanking_reads_vectorized
    
    def sum_inrepeat(metadata):
        inrepeat_reads = metadata['inrepeat_reads'].replace(' ', '').split('),(')
        inrepeat_reads_list = [tuple(map(int, t.strip('()').split(','))) for t in inrepeat_reads if t!= '']
        inrepeat_reads_sum = sum(t[1] for t in inrepeat_reads_list)
        return inrepeat_reads_sum
    
    def sum_spanning(metadata):
        spanning_reads = metadata['spanning_reads'].replace(' ', '').split('),(')
        spanning_reads_list = [tuple(map(int, t.strip('()').split(','))) for t in spanning_reads if t!= '']
        spanning_reads_sum = sum(t[1] for t in spanning_reads_list)
        return spanning_reads_sum