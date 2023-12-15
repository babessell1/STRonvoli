import os
import numpy as np
import pandas as pd
from pyfaidx import Fasta

def split_locus(locus:str):
    locus_list = locus.split('_')
    chr = locus_list[0]
    start = int(locus_list[1])
    end = int(locus_list[2])
    upstream = f'{chr}_{start-249}_{start+1}'
    downstream = f'{chr}_{end-1}_{end+249}'

    return upstream, downstream

def get_seq(locus:str, fasta):
    '''
    Params:
        locus: Genomic locus in "chrN_100_110" format (0-based)
        fasta: pyfaidx Fasta object using ref fasta of the same assembly as locus
    Returns:
        seq: DNA sequence at locus
    '''
    locus_list = locus.split('_')
    chr = locus_list[0]
    start = int(locus_list[1])
    end = int(locus_list[2])
    seq = fasta[chr][start:end].seq.upper()
    return seq

def one_hot_encode(seq:str):
    # https://stackoverflow.com/questions/34263772/how-to-generate-one-hot-encoding-for-dna-sequences
    mapping = dict(zip("ACGT", range(4)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2].astype(int)

def make_chr_size_dict(sizes_file):
    chr_size_dict = {}
    with open(sizes_file) as f:
        for line in f:
            line = line.strip().split('\t')
            if '_' not in line[0]:
                chr_size_dict[line[0]] = int(line[1])
    return chr_size_dict

if __name__ == "__main__":

    chr_sizes = '/nfs/turbo/dcmb-class/bioinf593/groups/group_05/refs/hg38.chrom.sizes'
    chr_size_dict = make_chr_size_dict(chr_sizes)

    depth_dir = '/nfs/turbo/dcmb-class/bioinf593/groups/group_05/output/depth'
    ohe_dir = '/nfs/turbo/dcmb-class/bioinf593/groups/group_05/STRonvoli/data/ohe'

    hg38 = Fasta('/nfs/turbo/boylelab/rintsen/genomes/hg38.fa')

    sample_names = ['HG01891', 'HG03492', 'HG02630', 'HG02257', 'NA19240', 'HG03098', 'HG01243']

    for filename in os.listdir(depth_dir):
        if 'all_repeat_regions' in filename or filename.split('_')[0] not in sample_names:
            continue
        else:
            with open(os.path.join(depth_dir, filename)) as pileup:
                for line in pileup:
                    line = line.strip().replace(', ',',').split(' ')
                    sample_name = line[0]

                    locus = line[1]
                    locus_list = locus.split('_')
                    chr = locus_list[0]
                    start = int(locus_list[1])
                    end = int(locus_list[2])

                    if (start-249)<0 or (end+249)>chr_size_dict[chr]: # Filter loci with windows that extend outside chromosome start/end
                        continue

                    up_locus, down_locus = split_locus(locus)
                    up_seq = get_seq(up_locus, hg38)
                    down_seq = get_seq(down_locus, hg38)
                    if 'N' in up_seq+down_seq:
                        continue
                    up_ohe = one_hot_encode(up_seq)
                    down_ohe = one_hot_encode(down_seq)
                    ohe = np.concatenate((up_ohe, down_ohe))

                    depths = np.atleast_2d(line[2].split(',')).astype(int).T
                    
                    if len(ohe) == len(depths):
                        final_mat = np.hstack((ohe, depths))
                        os.makedirs(os.path.join(ohe_dir, sample_name), exist_ok=True)
                        ohe_filepath = os.path.join(ohe_dir, sample_name, f'{sample_name}_{locus}.npy')
                        np.save(ohe_filepath, final_mat)