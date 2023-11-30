import os
import numpy as np
import pandas as pd
from torch.utils import data
import re

class STRDataset(data.Dataset):
    """
    Custom PyTorch dataset class mapping short-read STR loci and their corresponding features to long-read STR counts
    """
    def __init__(self, ohe_dir, metadata_file):
        self.ohe_dir = ohe_dir # each file is a one-hot encoding of the sequence and depth at a locus
        self.metadata = pd.read_csv(metadata_file, sep = '\t') # contains truth labels and metadata

    def __len__(self):
        return len(self.annots)
    
    def __getitem__(self, idx):
        locus = self.metadata['trid'].iloc[idx]
        sample_name = self.metadata['sample_name'].iloc[idx]

        ohe_file = os.path.join(self.ohe_dir, sample_name, f'{sample_name}_{locus}.npy')
        ohe = np.load(ohe_file)
        
        mc = self.metadata['MC'].iloc[idx]
        mc_split = mc.split(',')
        label = max(mc_split)
        
        metadata = self.metadata.iloc[idx]
   
        flanking_reads_vectorized = self.vectorize_flanking(metadata) # 50
        inrepeat_weighted_mean = self.inrepeat_weighted_mean(metadata) # 1
        spanning_reads_features = self.spanning_features(metadata) # 2
        motif_vec = self.vectorize_motif(metadata) # 24
        motif_len = metadata['len_motif'] # 1
        repeat_region_len = metadata['len_repeat_region'] # 1

        mlp_array = np.array([flanking_reads_vectorized, inrepeat_weighted_mean, spanning_reads_features, motif_vec, motif_len, repeat_region_len]) # len(mlp_array) == 79

        return ohe, label, mlp_array
    
    def vectorize_motif(self, metadata):
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
    
    def vectorize_flanking(self, metadata):
        flanking_reads = metadata['flanking_reads'].replace(' ', '').split('),(')
        if flanking_reads == ['()']:
            return np.zeros(50)
        else:
            flanking_reads_list = [tuple(map(int, t.strip('()').split(','))) for t in flanking_reads if t!= '']
            filtered_flanking_reads = [t for t in flanking_reads_list if t[0] <= 50] # number of loci with flanking reads in >50 repeats drastically drops off
            flanking_reads_vectorized = np.zeros(50)
            for index, value in filtered_flanking_reads:
                flanking_reads_vectorized[index - 1] = value
            return flanking_reads_vectorized
    
    def inrepeat_weighted_mean(self, metadata):
        '''Mean of repeats weighted by their inrepeat read depths'''
        inrepeat_reads = metadata['inrepeat_reads'].replace(' ', '').split('),(')
        if inrepeat_reads == ['()']:
            return 0
        else:
            inrepeat_reads_list = [tuple(map(int, t.strip('()').split(','))) for t in inrepeat_reads if t!= '']
            inrepeat_indices = [t[0] for t in inrepeat_reads_list]
            inrepeat_values = [t[1] for t in inrepeat_reads_list]
            inrepeat_weighted_mean = np.average(inrepeat_indices, weights=inrepeat_values)
            return inrepeat_weighted_mean
    
    def spanning_features(self, metadata):
        '''[Index, value] of the repeat count with largest depth'''
        spanning_reads = metadata['spanning_reads'].replace(' ', '').split('),(')
        if spanning_reads == ['()']:
            return np.zeros(2)
        else:
            spanning_reads_list = [tuple(map(int, t.strip('()').split(','))) for t in spanning_reads if t!= '']
            sorted_spanning_reads_list = sorted(spanning_reads_list, key=lambda x: x[1], reverse=True)
            spanning_reads_features = [item for t in sorted_spanning_reads_list[0] for item in t]
            return spanning_reads_features