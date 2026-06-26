import os
import sys
import multiprocessing

#from janggu.data import Bioseq, Cover
import pyBigWig
from pybedtools import BedTool
from Bio import SeqIO

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
import time

from sklearn import metrics, calibration
from itertools import product

from functools import partial
from itertools import repeat
from multiprocessing import Pool
import re
import subprocess

from typing import List

from MuRaL.data.distal_encoding import kmer_enc

def to_np(tensor):
    """Convert Tensor to numpy arrays"""
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()

def bed_reader_v0(bed_regions, central_bp):
    """
    Read a given BED region file and generate a new list of regions,
    and split the regions into two lists based on their strand information.

    Args:
    - bed_regions: BedTool object or bed file path representing the BED region file to read.
    - central_bp: Integer representing the length of the new regions to generate used encoding.

    Yields:
    Generator object that yields a list and a string:
    - The list contains regions used encoding.
    - The string represents the regions strand direction.
    """
    if isinstance(bed_regions, str):
        bed_regions = BedTool(bed_regions)
    else:
        if not isinstance(bed_regions, BedTool):
            print(f"Error: bed_regions should be <str> or <Bedtools>, but input is {bed_regions.__class__}!")
            sys.exit()
    init = 0
    for region in bed_regions:
        if not init:
            init += 1
            chrom, start0 = str(region.chrom), region.start
            end0 = start0 + central_bp 
            pos_strand_region = []
            neg_strand_region = []
            
        chrom2, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand
            
        if chrom2 != chrom:
            if pos_strand_region:
                yield pos_strand_region, '+'
                pos_strand_region = []
            if neg_strand_region:
                yield neg_strand_region, '-'
                neg_strand_region = []    
            chrom = chrom2
            start0 = 1
            end0 = 1 + central_bp 
                        
        if strand == '+':
            pos_strand_region.append(region)

        else:
            neg_strand_region.append(region)
            
        if start > end0:
            if pos_strand_region:
                yield pos_strand_region, '+'
                pos_strand_region = []

            if neg_strand_region:
                yield neg_strand_region, '-'
                neg_strand_region = []

            start0 = end0
            end0 += central_bp
            
    if pos_strand_region:
        yield pos_strand_region, '+'
    if neg_strand_region:
        yield neg_strand_region, '-'

def bed_reader(bed_regions, central_bp):
    """
    Read a given BED region file and generate a new list of regions,
    and split the regions into two lists based on their strand information.

    Args:
    - bed_regions: BedTool object or bed file path representing the BED region file to read.
    - central_bp: Integer representing the length of the new regions to generate used encoding.

    Yields:
    Generator object that yields a list and a string:
    - The list contains regions used encoding.
    - The string represents the regions strand direction.
    """
    if isinstance(bed_regions, str):
        bed_regions = BedTool(bed_regions)
    else:
        if not isinstance(bed_regions, BedTool):
            print(f"Error: bed_regions should be <str> or <Bedtools>, but input is {bed_regions.__class__}!")
            sys.exit()
    init = 0
    for region in bed_regions:
        if not init:
            init += 1
            chrom, start0 = str(region.chrom), region.start
            end0 = start0 + central_bp 
            pos_strand_region = []
            neg_strand_region = []
            
        chrom2, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand
            
        if chrom2 != chrom:
            if pos_strand_region:
                yield pos_strand_region, '+'
                pos_strand_region = []
            if neg_strand_region:
                yield neg_strand_region, '-'
                neg_strand_region = []    
            chrom = chrom2
            start0 = 1
            end0 = 1 + central_bp 
            

            
        if start > end0:
            while start > end0:
                start0 = end0
                end0 += central_bp
            if pos_strand_region:
                yield pos_strand_region, '+'
                pos_strand_region = []

            if neg_strand_region:
                yield neg_strand_region, '-'
                neg_strand_region = []


        if strand == '+':
            pos_strand_region.append(region)

        else:
            neg_strand_region.append(region)          

    if pos_strand_region:
        yield pos_strand_region, '+'
    if neg_strand_region:
        yield neg_strand_region, '-'

def get_position_info(test_bed, central_radius):
    """
    Get validation position information

    Returns:
    pd.DataFrame: A DataFrame containing the chromosome, start, end, and strand
                  information with columns ['chrom', 'start', 'end', 'strand'].
    """
    bed_generator = bed_reader(test_bed, central_radius)
    info = []
    for batch, strand in bed_generator:
        info.extend([[nucleo.chrom, nucleo.start, nucleo.end, nucleo.name, strand, nucleo.score] for nucleo in batch])
    info = pd.DataFrame(info, columns=['chrom', 'start', 'end', 'info', 'strand', 'mut_type'])
    return info

def get_position_info_by_trainset(test_bed, central_radius):
    """
    Get validation position information from the test bed file.

    This function reads a BED file and processes its entries in batches.
    It returns a MultiIndex DataFrame where the first level index is the 
    batch number and the second level index is the sample number within each batch.

    Returns:
    pd.DataFrame: A DataFrame containing the chromosome, start, end, and strand 
                  information with a MultiIndex (batch_num, sample_num).
    """
    bed_generator = bed_reader(test_bed, central_radius)
    info = []

    for batch_num, (batch, stand) in enumerate(bed_generator):
        batch_info = [[batch_num, i, nucleo.chrom, nucleo.start, nucleo.end, stand] for i, nucleo in enumerate(batch)]
        info.extend(batch_info)
    
    info_df = pd.DataFrame(info, columns=['batch_num', 'sample_num', 'chrom', 'start', 'end', 'stand'])
    info_df.set_index(['batch_num', 'sample_num'], inplace=True)
    
    return info_df

def get_bw_for_bed(bw_files, bed_regions, radius):

    bw_fh = []
    for file in bw_files:
        bw_fh.append(pyBigWig.open(file))
    
    bw_data = []
    
    seq_len = radius*2+1
    
    if len(bw_fh) > 0:
        
        for i, region in enumerate(bed_regions):
            chrom, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand
            #bw_values = []
            #seq_len = [bw.chroms(chrom) for bw in bw_fh]
            
            bw_list = []
            for j, bw in enumerate(bw_fh):
                            
                start1 = max([int(start)-radius, 0])
                stop1 = min([int(stop)+radius, bw.chroms(chrom)])
                
                bw_values = np.nan_to_num(bw.values(chrom, start1, stop1, numpy=True))
                if(len(bw_values) < seq_len):
                    if start1 == 0:
                        bw_values = np.concatenate([(seq_len - len(bw_values))*[0], bw_values])
                    else:
                        bw_values = np.concatenate([bw_values, (seq_len - len(bw_values))*[0]])
                
                if strand == '-':
                    bw_values = np.flip(bw_values)
                
                bw_list.append(bw_values)
            
            bw_data.append(bw_list)     

        bw_data = np.array(bw_data).astype(np.float32)
    
    return bw_data

def open_bigwig_files(bw_files):
    return [pyBigWig.open(file) for file in bw_files]


def get_bw_data(bw_fh, chrom, start, stop, strand, long_seq_len, bw_files_len, end):
    # Initialize optional imputation arrays
    annot_left_impute = None
    annot_right_impute = None

    # Handle left-side imputation if start is negative
    if start < 0:
        left_impute = -start
        start = 0  # Adjust start to the beginning of the sequence
        annot_left_impute = np.zeros((bw_files_len, left_impute))  # Left padding with zeros

    # Handle right-side imputation if needed
    if end:
        right_impute = stop - long_seq_len
        annot_right_impute = np.zeros((bw_files_len, right_impute))  # Right padding with zeros
        stop = long_seq_len  # Adjust stop to the max sequence length

    # Read BigWig data and replace NaNs with 0
    bw_data = np.asarray([np.nan_to_num(bw.values(chrom, start, stop, numpy=True)) for bw in bw_fh])

    # Concatenate left-side imputation if applicable
    if annot_left_impute is not None:
        bw_data = np.concatenate([annot_left_impute, bw_data], axis=1)

    # Concatenate right-side imputation if applicable
    if annot_right_impute is not None:
        bw_data = np.concatenate([bw_data, annot_right_impute], axis=1)

    return bw_data

def segment_annot(long_seq_len, bw_fh, start, stop, chrom, strand, end=False):
    bw_data = get_bw_data(bw_fh, chrom, start, stop, strand, long_seq_len, len(bw_fh), end)
    return bw_data

def batch_annot_by_segment(bw_data, index, radius):
    window_size = 2 * radius + 1  
    return np.asarray([bw_data[:, start1: start1 + window_size] for start1 in index], dtype=float)

def annot_encoding_by_region(bw_fh, seqs, batch_shape, radius, seq_records):
    if not hasattr(seqs, '__iter__'):
        sys.exit("Error: input seqs is not <generator>!")
    
    batch_annot_encoding = np.empty((batch_shape, len(bw_fh), 2 * radius + 1), dtype='float32')
    batch_index = 0
    long_seq_len = None
    
    for start0, stop0, chrom, strand, index, end in seqs:
        if long_seq_len is None:
            long_seq_len = len(str(seq_records[chrom].seq))
        
        bw_data = segment_annot(long_seq_len, bw_fh, start0, stop0, chrom, strand, end)
        annot_seq = batch_annot_by_segment(bw_data, index, radius)

        sub_batch_num = len(index)
        batch_annot_encoding[batch_index:batch_index + sub_batch_num] = annot_seq
        batch_index += sub_batch_num
    
    return batch_annot_encoding
#########################################################################
#                          gene HD5
#  use HD5 for saving distal Encoding(One-Hot)
#########################################################################
def get_h5f_path(bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal):
    """Get the H5 file path name based on input data"""
    
    h5f_path = bed_file + '.distal_' + str(distal_radius) + '.segment_' + str(central_radius) + '_segshare'
    
    if distal_order > 1:
        h5f_path = h5f_path + '_' + str(distal_order)
        
    if len(bw_names) > 0 and (not without_bw_distal):
        h5f_path = h5f_path + '.' + '.'.join(list(bw_names))
    
    h5f_path = h5f_path + '.h5'
    
    return h5f_path

def change_h5f_path(h5f_path, bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal):
    name = get_h5f_path(bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal)
    name = name.split('/')[-1]
    h5f_path_new = os.path.join(h5f_path, name) 

    if not os.path.isdir(h5f_path):
        print(f"Warming : input h5f_path not dir, h5f path generate to {h5f_path_new} !")
    return h5f_path_new

def generate_h5f(bed_regions, h5f_path, ref_genome, central_radius, distal_radius, distal_order, bw_files, h5_chunk_size=1, chunk_size=50000, without_bw_distal=True):
    """Generate the H5 file for storing distal data"""

    print('Generating HDF5 file:', h5f_path)
    sys.stdout.flush()
    # recode overlap realtion ship of sample in each segment
    with open(ref_genome, 'r') as f:
        seq_records = SeqIO.to_dict(SeqIO.parse(f, 'fasta'))
    seq_list, batch_shape = get_distal_seqs_by_region(bed_regions, seq_records, distal_radius, central_radius)
    
    with h5py.File(h5f_path, 'w') as hf:
        for idx in range(len(batch_shape)):
            # Creat group And Wirte in HD5 for each segment
            create_group_and_store(hf, idx, seq_list[idx], seq_records, distal_radius)
    
    return None

def create_group_and_store(hf, idx, seqs_info, seq_records, distal_radius):
    """Create a group for each segment and store encoded data"""
    group = hf.create_group(f'segment_{idx}')
    stand = seqs_info[0][3]
    group.attrs['stand'] = stand
    # Split segment into multi sample share sub-segment 
    iter_segment_info = get_segment_info(seqs_info, seq_records, distal_radius)
    
    store_encodings(group, iter_segment_info)

def store_encodings(group, iter_segment_info):
    """Store encoding sample share sub-segment into the HDF5 group"""
    for sample_num, (long_seq, start, stop, chrom, strand, radius, index, end) in enumerate(iter_segment_info):
        encoding = segment_ohe_encoder(long_seq, start, stop, strand, radius, end)
        sample_dset = group.create_dataset(f'sample_{sample_num}', data=encoding, compression="gzip", compression_opts=4)
        sample_dset.attrs['index'] = index # recode sample index in sub-segment

def get_segment_info(seqs, seq_records, radius):
    init = True
    for start0, stop0, chrom, strand, index, end in seqs:
        if init:
            init = False
            long_seq = str(seq_records[chrom].seq)
            c = chrom
        assert chrom == c
        yield long_seq, start0, stop0, chrom, strand, radius, index, end

def segment_ohe_encoder(long_seq, start, stop, strand, radius, end=False):

    one_hot_encoder = {'A':np.array([[1,0,0,0]], dtype=np.float32).T,
               'C':np.array([[0,1,0,0]], dtype=np.float32).T,
               'G':np.array([[0,0,1,0]], dtype=np.float32).T,
               'T':np.array([[0,0,0,1]], dtype=np.float32).T,
               'R':np.array([[0.5,0,0.5,0]], dtype=np.float32).T, #A,G
               'Y':np.array([[0,0.5,0,0.5]], dtype=np.float32).T, #C,T
               'M':np.array([[0.5,0.5,0,0]], dtype=np.float32).T, #A,C
               'S':np.array([[0,0.5,0.5,0]], dtype=np.float32).T, #C,G
               'W':np.array([[0.5,0,0,0.5]], dtype=np.float32).T, #A,T
               'K':np.array([[0,0,0.5,0.5]], dtype=np.float32).T, #G,T
               'B':np.array([[0,1/3,1/3,1/3]], dtype=np.float32).T, #not A
               'D':np.array([[1/3,0,1/3,1/3]], dtype=np.float32).T, #not C
               'H':np.array([[1/3,1/3,0,1/3]], dtype=np.float32).T, #not G
               'V':np.array([[1/3,1/3,1/3,0]], dtype=np.float32).T, #not T
               'N':np.array([[0.25,0.25,0.25,0.25]], dtype=np.float32).T}

    one_hot_encoder_rc = {'A':np.array([[0,0,0,1]], dtype=np.float32).T,
               'C':np.array([[0,0,1,0]], dtype=np.float32).T,
               'G':np.array([[0,1,0,0]], dtype=np.float32).T,
               'T':np.array([[1,0,0,0]], dtype=np.float32).T,
               'R':np.array([[0,0.5,0,0.5]], dtype=np.float32).T, #A,G
               'Y':np.array([[0.5,0,0.5,0]], dtype=np.float32).T, #C,T
               'M':np.array([[0,0,0.5,0.5]], dtype=np.float32).T, #A,C
               'S':np.array([[0,0.5,0.5,0]], dtype=np.float32).T, #C,G
               'W':np.array([[0.5,0,0,0.5]], dtype=np.float32).T, #A,T
               'K':np.array([[0.5,0.5,0,0]], dtype=np.float32).T, #G,T
               'B':np.array([[1/3,1/3,1/3,0]], dtype=np.float32).T, #not A
               'D':np.array([[1/3,1/3,0,1/3]], dtype=np.float32).T, #not C
               'H':np.array([[1/3,0,1/3,1/3]], dtype=np.float32).T, #not G
               'V':np.array([[0,1/3,1/3,1/3]], dtype=np.float32).T, #not T
               'N':np.array([[0.25,0.25,0.25,0.25]], dtype=np.float32).T}
        
    #imput 
    short_seq = ['', '','']
    if start < 0:
        left_impute = 0 - start 
        start = 0
        short_seq[0] = left_impute * 'N'
        
    if end:
        long_seq_len = len(long_seq)
        right_impute = stop - long_seq_len
        short_seq[2] = right_impute * 'N'

    short_seq[1] = long_seq[start:stop].upper()

    short_seq = ''.join(short_seq)

    
   # return short_seq
    if strand == '+':
        distal_ecoding = np.concatenate([one_hot_encoder[c] for c in short_seq], axis=1)
        
    else:
        distal_ecoding = np.concatenate([one_hot_encoder_rc[c] for c in short_seq[::-1]], axis=1)
        
    return distal_ecoding

def generate_h5fv2(bed_regions, h5f_path, ref_genome, central_radius, distal_radius, distal_order, bw_paths, bw_files, chunk_size=50000, n_h5_files=1, without_bw_distal=False):
    """Generate the H5 file for storing distal data"""
    write_h5f = True
    if os.path.exists(h5f_path):
        try:
            with h5py.File(h5f_path, 'r', swmr=True) as hf:
                bed_path = bed_regions.fn
                h5_sample_size = check_h5f_sample_size(hf)
                try:
                    if os.lstat(bed_path).st_mtime < os.lstat(h5f_path).st_mtime \
                    and len(bed_regions) == h5_sample_size:
                        write_h5f = False
                except KeyError:
                    print('Warning: re-genenerating the H5 file, because the file is empty or imcomplete:', h5f_path)
                                       
        except OSError:
            print('Warning: re-genenerating the H5 file, because the file is empty or imcomplete:', h5f_path)

            
    # If the H5 file is unavailable or incomplete, generate the file
    if write_h5f:            
        p = multiprocessing.Process(target=generate_h5f,\
                                    args=(bed_regions,h5f_path,ref_genome,central_radius, \
                                          distal_radius,distal_order,bw_files)\
                                   )       
    
        return p
    return 0

def check_h5f_sample_size(h5f):
    h5_sample_size = 0
    for key in h5f.keys():
        segment = h5f[key]
        h5_sample_size += sum([len(segment[key].attrs['index']) for key in segment.keys()])
    return h5_sample_size

#########################################################################
#                           Local Embeding 
#########################################################################
def prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_bp, local_radius, local_order, seq_only):
    """Prepare local data for given regions"""
    """  
        Args:
            bed_regions: <Bedtools> 
            ref_genome:  <SeqRecord> ref genome
    """
    # Read the seq data
    local_seq_cat, y = local_digitalized_seqs_by_region(bed_regions, ref_genome, central_bp=central_bp, local_radius=local_radius, local_order=1)    
    #local_seq_cat = pd.concat([pd.DataFrame(x,columns = seq_cols) for x in local_seq_cat],keys=range(len(local_seq_cat)))
    local_seq_cat = pd.concat(local_seq_cat,keys=range(len(local_seq_cat)))

    seq_cols = ['us'+str(local_radius - i) for i in range(local_radius)] + ['mid'] + ['ds'+str(i+1) for i in range(local_radius)]
    local_seq_cat = pd.DataFrame(local_seq_cat, columns = seq_cols)
    if local_order > 1:
        local_seq_cat2, y = local_digitalized_seqs_by_region(bed_regions, ref_genome, central_bp, local_radius, local_order=local_order)
        
        # NOTE: replace k-mers with 'N' with a large number; the padding numbers at the two ends of the chromosomes are also large numbers   
        #local_seq_cat2 = np.where(np.logical_and(local_seq_cat2>=0, local_seq_cat2<=4**local_order), local_seq_cat2, 4**local_order)
        # Names of the categorical variables
        cat_n = local_radius*2 +1 - (local_order-1)
        categorical_features  = ['cat'+str(i+1) for i in range(cat_n)]
        #local_seq_cat = pd.concat([pd.DataFrame(x,columns = categorical_features) for x in local_seq_cat2],keys=range(len(local_seq_cat)))
        local_seq_cat2 = pd.concat(local_seq_cat2,keys=range(len(local_seq_cat2)))
        local_seq_cat2 = pd.concat([local_seq_cat, local_seq_cat2], axis=1)
    else:
        categorical_features = seq_cols
    
    print('local_seq_cat2 shape and columns:', local_seq_cat2.shape, local_seq_cat2.columns)
    print('categorical_features:', categorical_features)
    
    # The 'score' field in the BED file stores the label/class information
    y = pd.concat(y,keys=range(len(y)))
    output_feature = 'mut_type'
    
    # Add feature data in bigWig files
    # bug seq_only = false , concat multi Index dataframe and Index dataframe
    seq_only = True
    if len(bw_files) > 0 and seq_only == False:
        get_local_bw_data_by_region(bw_files, bw_names, bw_radii, bed_regions, central_bp=central_bp)
        # Use the mean value of the region of 2*radius+1 bp around the focal site
        bw_data = get_mean_bw_for_bed(bw_files, bw_names, bw_radii, bed_regions)
        data_local = pd.concat([local_seq_cat2, bw_data, y], axis=1)
    else:
        data_local = pd.concat([local_seq_cat2, y], axis=1)

    return data_local, seq_cols, categorical_features, output_feature

# def prepare_local_datav2(bed_regions,ref_genome, bw_files, bw_names, bw_radii, central_bp, local_radius, local_order, seq_only):

#     bed_generator = bed_reader(bed_regions, central_bp)
#     seqs_information_generator = seq_generator(bed_generator, ref_records=ref_genome, local_radius=local_radius)
#     y = []
#     local_seq_cat = []
#     local_seq_cat2 = []

#     bw_data = []
#     # Check if bigWig data should be used
#     use_bigwig = len(bw_files) > 0 and not seq_only

#     for long_seq, seqs, label in seqs_information_generator:
#         sample_number = len(label)
#         local_seq_cat_by_region = kmer_encoding_by_seqs(long_seq, seqs, sample_number, local_radius=local_radius, local_order=1)
#         local_seq_cat.append(local_seq_cat_by_region)
        
#         if local_order > 1:
#             local_seq_cat_by_region = kmer_encoding_by_seqs(long_seq, seqs, sample_number, local_radius=local_radius, local_order=local_order)
#             local_seq_cat2.append(local_seq_cat_by_region)
#         if use_bigwig:
#             bw_data_by_region = get_local_bw_data_by_region(bw_files=bw_files, bw_radii=bw_radii, bw_names=bw_names,sample_number=sample_number, seqs=seqs)
#             bw_data.append(bw_data_by_region)

#         y.append(pd.DataFrame(label.reshape((-1,1)),columns = ['mut_type']))
    
#     if local_seq_cat:
#         local_seq_cat = pd.concat(local_seq_cat, keys=range(len(local_seq_cat)))
#         seq_cols = ['us'+str(local_radius - i) for i in range(local_radius)] + ['mid'] + ['ds'+str(i+1) for i in range(local_radius)]
#     else:
#         raise ValueError("local_seq_cat is empty. Ensure that sequences are being generated correctly.")

#     if local_seq_cat2:
#         cat_n = local_radius*2 +1 - (local_order-1)
#         categorical_features  = ['cat'+str(i+1) for i in range(cat_n)]

#         local_seq_cat2 = pd.concat(local_seq_cat2, keys=range(len(local_seq_cat2)))
#         local_seq_cat2 = pd.concat([local_seq_cat, local_seq_cat2], axis=1)
#     else:
#         categorical_features = seq_cols

#     print('local_seq_cat2 shape and columns:', local_seq_cat2.shape, local_seq_cat2.columns)
#     print('categorical_features:', categorical_features)

#     y = pd.concat(y, keys=range(len(y)))
#     col_name_label = 'mut_type'

#     # Add feature data in bigWig files
#     # bug seq_only = false , concat multi Index dataframe and Index dataframe
#     if bw_data:
#         bw_data = pd.concat(bw_data, keys=range(len(bw_data)))
#         # Use the mean value of the region of 2*radius+1 bp around the focal site
#         data_local = pd.concat([local_seq_cat2, bw_data, y], axis=1)
#     else:
#         data_local = pd.concat([local_seq_cat2, y], axis=1)

#     return data_local, seq_cols, categorical_features, col_name_label

def prepare_local_feature(segments, ref_genome, local_radius, local_order, names=None):

    # bed_generator = bed_reader(bed_regions, central_segment)
    seqs_information_generator = seq_generator(segments, ref_records=ref_genome, local_radius=local_radius)
    y = []
    w = []
    local_seq = []
    local_seq_encode = []

    # Check if bigWig data should be used

    for long_seq, seqs, label, weight in seqs_information_generator:
        sample_number = len(label)
        local_seq_cat_by_region = kmer_encoding_by_seqs(long_seq, seqs, sample_number, local_radius=local_radius, local_order=1)
        # .values : convert DF to np.array, backward compatibility with previous code
        local_seq.append(local_seq_cat_by_region.values)

        if local_order > 1:
            local_seq_cat_by_region = kmer_encoding_by_seqs(long_seq, seqs, sample_number, local_radius=local_radius, local_order=local_order)
            # .values : convert DF to np.array, backward compatibility with previous code
            local_seq_encode.append(local_seq_cat_by_region.values)

        # y.append(pd.DataFrame(label.reshape((-1,1)),columns = ['mut_type']))
        y.append(label)
        w.append(weight)

    if local_seq:
        seq_cols = ['us'+str(local_radius - i) for i in range(local_radius)] + ['mid'] + ['ds'+str(i+1) for i in range(local_radius)]
        local_seq = local_seq # (segment, sample_number, 2*local_radius+1), non-regular nested, each segment has different sample_number
    else:
        raise ValueError("local_seq is empty. Ensure that sequences are being generated correctly.")

    if local_seq_encode:
        cat_n = local_radius*2 +1 - (local_order-1)
        categorical_features  = ['cat'+str(i+1) for i in range(cat_n)]
        local_seq_encode = local_seq_encode # (segment, sample_number, cat_n), non-regular nested, each segment has different sample_number
    else:
        categorical_features = seq_cols

    print('segment number:', len(local_seq_encode))
    print('categorical_features:', categorical_features)

    # col_name_label = 'mut_type'

    if names is None:
        names = ['local_seq', 'local_seq_encode', 'mut_type']
    return {
        names[0]: local_seq,
        names[1]: local_seq_encode,
        names[2]: y,
        'sample_weight': w,
    }



def kmer_encoding_by_seqs(long_seq, seqs, sample_number, local_radius, local_order):
    """Region as the minimum unit"""
    cat_n = local_radius*2 +1 - (local_order-1) 
    outlier_process = preocess_local_seq_outlier(local_order, local_radius)

    if local_order == 1:
        seq_cols = ['us'+str(local_radius - i) for i in range(local_radius)] + ['mid'] + ['ds'+str(i+1) for i in range(local_radius)]
    else:
        seq_cols  = ['cat'+str(i+1) for i in range(cat_n)]

    kmer_encoding_seqs = np.empty((sample_number, cat_n),dtype=np.int64)

    kmer_encoding_seqs = local_encoding_seqs(long_seq, seqs, local_radius, kmer_encoding_seqs, local_order=local_order)
    kmer_encoding_seqs = outlier_process(kmer_encoding_seqs)
    kmer_encoding_seqs = pd.DataFrame(kmer_encoding_seqs,columns=seq_cols)

    return kmer_encoding_seqs


def seq_generator(bed_generator, ref_records, local_radius):
    init = False
    for batch, stand in bed_generator:
        if not init:
            chrom = batch[0].chrom
            long_seq = str(ref_records[chrom].seq)
            init = True
        else:
            if chrom != batch[0].chrom:
                chrom = batch[0].chrom
                long_seq = str(ref_records[chrom].seq)

        seqs = list(get_seqs_to_digitalized(long_seq, batch, local_radius, stand))
        label = get_label(batch)
        weight = get_weight(batch)
        yield long_seq, seqs, label, weight


def local_digitalized_seqs_by_region(bed_regions, seq_records, central_bp, local_radius, local_order=1):

    if 'items' not in dir(seq_records):
        _ = type(seq_records)
        sys.exit(f'seq_records need be dict, but input is {_} !')
    
    cat_n = local_radius*2 +1 - (local_order-1) 
    outlier_process = preocess_local_seq_outlier(local_order, local_radius)
    
    if local_order == 1:
        seq_cols = ['us'+str(local_radius - i) for i in range(local_radius)] + ['mid'] + ['ds'+str(i+1) for i in range(local_radius)]
        
    else:
        seq_cols  = ['cat'+str(i+1) for i in range(cat_n)]
    
    bed_generator = bed_reader(bed_regions, central_bp)
    digit_dataset = []
    y = []
    init = False
    for batch,stand in bed_generator:
        if not init:
            chrom = batch[0].chrom
            long_seq = str(seq_records[chrom].seq)
            init = True
        else:
            if chrom != batch[0].chrom:
                chrom = batch[0].chrom
                long_seq = str(seq_records[chrom].seq)

        batch_local_encoding = np.empty((len(batch),cat_n), dtype=np.int64)
        seqs = get_seqs_to_digitalized(long_seq, batch, local_radius, stand)
        digit_seqs = local_encoding_seqs(long_seq, seqs, local_radius, batch_local_encoding, local_order=local_order)
        digit_seqs = outlier_process(digit_seqs)
        digit_dataset.append(pd.DataFrame(digit_seqs,columns=seq_cols))

        label = get_label(batch)
        y.append(pd.DataFrame(label.reshape((-1,1)),columns = ['mut_type']))


    #digit_dataset = np.concatenate(digit_dataset)
    return digit_dataset,y 

def preocess_local_seq_outlier(local_order,local_radius):
    """
    Generate a function to process local sequence outliers based on the given local order and radius.

    Returns:
    A function that processes local sequence outliers.

    Raises:
    - SystemExit: If local_order is not greater than 0.
    """
    if local_order == 1:
        def process_local_seq(local_seq_cat,local_radius=local_radius):
            """
            Process the local sequence for local order 1.
            """
            if np.unique(local_seq_cat[:,local_radius], axis=0).shape[0] != 1:
                print('ERROR: The positions in input BED file have different bases (A/T and C/G mixed)! The ref_genome or input BED file could be wrong.', file=sys.stderr)
                sys.exit()
            return np.where(local_seq_cat>=0, local_seq_cat, 0)
        return process_local_seq
    
    elif local_order > 1:
        v = 4**local_order
        def process_local_seq(local_seq_cat,v=v):
            """
            Process the local sequence for local order greater than 1.
            """
            return np.where(np.logical_and(local_seq_cat >= 0, local_seq_cat <= v), local_seq_cat, v)
        return process_local_seq
    else:
        sys.exit("local_oreder need larger than 0!")

def get_seqs_to_digitalized(long_seq, regions, radius, seq_strand):
    """
    Check relationship of regions by radius and save the information for encoding.

    Yields:
    A tuple containing start position of encoding, stop position of encoding, chromosome,
             strand, index list, and a boolean indicating if imputation is needed
    - The index list used to find samples from encoding segment.

    """
    #seqs = []
    end=True
    init = False
    index = [0]
    for region in regions:
        chrom, start, stop, strand = str(region.chrom), region.start, region.stop, region.strand
        
        assert seq_strand == strand
        
        # init 
        if not init:
            init = True
            chrom_init = chrom
            start0 = int(start) - radius
            stop0 = int(stop) + radius
            leng_seq = len(long_seq)
            continue
            
        start1 = int(start) - radius
        stop1 = int(stop) + radius
        
        if start1 > stop0:
            # one-hot encoding
            impute = False
            yield (start0, stop0, chrom, strand, index, impute)
            start0, stop0 = start1, stop1
            index = [0]
        else:
            stop0 = stop1
            index.append(start1-start0)
    if stop0 > leng_seq:
        impute = True
    else:
        impute = False
    yield (start0, stop0, chrom, strand, index, impute)

def local_encoding_seqs(long_seq, seqs, radius, batch_local_encoding, local_order):
    if '__iter__' not in dir(seqs):
        sys.exit("Error : input seqs is not <generator>!")

    if not isinstance(batch_local_encoding, np.ndarray):
        sys.exit("Error: one-hot Encoding Need provided an array to save batch infomation!")
        
    batch_index = 0
    for start0, stop0, chrom, strand, index, end in seqs:
        sub_batch_num = len(index)
        sub_batch = seq_digit_encoder(long_seq, start0, stop0, chrom, strand, radius,index, local_order, end)
        batch_local_encoding[batch_index:batch_index+sub_batch_num] = sub_batch
        batch_index += sub_batch_num
    return batch_local_encoding 

def seq_digit_encoder(long_seq, start, stop, chrom, strand, radius, index, local_order, end=False):
    digit_encoder = {'A':0,'C':1,'G':2,'T':3,
               'R':-1, #A,G
               'Y':-1, #C,T
               'M':-1, #A,C
               'S':-1, #C,G
               'W':-1, #A,T
               'K':-1, #G,T
               'B':-1, #not A
               'D':-1, #not C
               'H':-1, #not G
               'V':-1, #not T
               'N':-1}

    digit_encoder_rc = {'A':3, 'C':2, 'G':1, 'T':0,
               'R':-1, #A,G
               'Y':-1, #C,T
               'M':-1, #A,C
               'S':-1, #C,G
               'W':-1, #A,T
               'K':-1, #G,T
               'B':-1, #not A
               'D':-1, #not C
               'H':-1, #not G
               'V':-1, #not T
               'N':-1}
    
    #impute
    short_seq = ['', '','']
    if start < 0:
        #left_imput = 0 - start + 1
        left_impute = 0 - start
        start = 0
        short_seq[0] = left_impute * 'N'
        
    if end:
        long_seq_len = len(long_seq)
        right_impute = stop - long_seq_len
        short_seq[2] = right_impute * 'N'

    short_seq[1] = long_seq[start:stop].upper()
    short_seq = ''.join(short_seq)
  #  return short_seq
    if strand == '+':
        digit_seq = np.array([digit_encoder[c] for c in short_seq])
    else:
        digit_seq = np.array([digit_encoder_rc[c] for c in short_seq[::-1]])

    if local_order > 1:
        seq_len = len(digit_seq)
        new_seq = []
        for i in range(seq_len - local_order +1):
            kmer = digit_seq[i:i+local_order]
            if min(kmer) < 0:
                new_seq.append(-1)
            else:
                digit = sum([kmer[d]*4**(local_order-d-1) for d in range(local_order)])
                new_seq.append(digit)
        digit_seq = np.array(new_seq)
        window_size = 2 * radius + 1 - local_order + 1
    else:
        window_size = 2 * radius + 1

    if strand == '+':
        digit_seq = np.asarray([digit_seq[start1:start1 + window_size] for start1 in index], dtype=np.int64)         
    else:
        digit_seq = np.asarray([digit_seq[-start1-window_size:-start1] if start1 else digit_seq[-start1-window_size:] for start1 in index], dtype=np.int64)
    
    digit_seq = np.where(np.logical_and(digit_seq>=0, digit_seq<=4**local_order), digit_seq, 4**local_order)
    return digit_seq

def get_local_bw_data_by_region(bw_files:List[str],
                                bw_radii, 
                                bw_names,
                                sample_number,
                                seqs):
    bw_fh = [pyBigWig.open(file) for file in bw_files]
    batch_bw_data = np.zeros((sample_number, len(bw_fh)),dtype=float)
    batch_bw_data = get_mean_bw_seqs(bw_fh, seqs, bw_radii, batch_bw_data)
    batch_bw_data = pd.DataFrame(batch_bw_data, columns=bw_names)
    return batch_bw_data


def get_mean_bw_seqs(bw_fh, seqs, bw_radii, batch_bw_data):

    batch_index = 0
    for start, stop, chrom, strand, index, end in seqs:
        sub_batch_num = len(index)
        sub_batch = get_mean_bw_base_each_pointend(bw_fh, start, stop, chrom, index, bw_radii)
        batch_bw_data[batch_index:batch_index+sub_batch_num] = sub_batch
        batch_index += sub_batch_num
    
    return batch_bw_data 


def get_mean_bw_base_each_pointend(bw_fh, start, stop, chrom, index, bw_radii):
    mean_bw_seq = []
    for i, bw in enumerate(bw_fh):
        window_size = bw_radii[i] * 2 + 1
        each_bw_seq = np.nan_to_num(bw.values(chrom, start, stop))
        each_mean_bw_seq = np.asarray([
            np.mean(each_bw_seq[start: min(start+window_size, len(each_bw_seq))]) 
            for start in index
        ], dtype=np.float)
        mean_bw_seq.append(each_mean_bw_seq)
    # shape: (bw_name, sample)
    mean_bw_seq = np.asarray(mean_bw_seq)
    # reshape: (sample, bw_name)
    mean_bw_seq = mean_bw_seq.T

    return mean_bw_seq


def get_mean_bw_for_bed(bw_fh, bw_radii, bed_regions):
    
    bw_data = np.zeros((len(bed_regions), len(bw_fh)), dtype=float)
    
    if len(bw_fh) > 0:
        
        for i, region in enumerate(bed_regions):
            chrom, start, stop = str(region.chrom), region.start, region.stop
            #bw_values = []
            #seq_len = [bw.chroms(chrom) for bw in bw_fh]
            
            for j, bw in enumerate(bw_fh):
                            
                start1 = max([int(start)-bw_radii[j], 0])
                stop1 = min([int(stop)+bw_radii[j], bw.chroms(chrom)])
                bw_data[i,j] = np.nan_to_num(bw.values(chrom, start1, stop1, numpy=True)).mean()
                

        bw_data = pd.DataFrame(bw_data, columns=bw_names)
    
    return bw_data 

def get_label(bed_regions):
    y = np.array([float(loc.score) for loc in bed_regions])
    return y

def get_weight(bed_regions):
    """Extract per-site count from BED name field (col4).
    Format: 'chr1:238329;G>A;-1;1' — last field after ';' is count.
    Non-mutated sites have name='.' → weight=1.0
    """
    weights = []
    for loc in bed_regions:
        if loc.name in ('.', '', 'na'):
            weights.append(1.0)
        else:
            try:
                count = float(loc.name.split(';')[-1])
                weights.append(count)
            except (ValueError, IndexError):
                weights.append(1.0)
    return np.array(weights, dtype=np.float32)
#########################################################################
#                          Construct Dataset Without HDF5 
# 
# Note: When sample redundancy is high, I/O in preprocessing can become a bottleneck
#       in model training. This method computes distal encoding dirctly from the reference  
#       genome, reducing the disk IO. 
#
# Suggestion: For human data, if the distal region is greater than 8k,
#             it is recommended to use this method.
#########################################################################
def max_min_norm(data):
    data = np.asarray(data)
    v_max, v_min = np.max(data), np.min(data)
    return (data - v_min) / (v_max - v_min)





def prepare_dataset_np(bed_regions, ref_genome, bw_files, bw_names, bw_radii,central_radius=30000, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False, without_bw_distal=False, segment_task=None):
    """Prepare the datasets for given regions, without an H5 file"""
    """  
        Args:
            bed_regions: <Bedtools> 
            ref_genome:  <str> path of ref genome
    """
    # Prepare local data
    with open(ref_genome, 'r') as f:
        ref_genome = SeqIO.to_dict(SeqIO.parse(f, 'fasta'))
    data_local, seq_cols, categorical_features, output_feature = prepare_local_datav2(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNP(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, central_radius=central_radius, distal_radius=distal_radius, n_channels=n_channels, bw_files=bw_files, seq_only=seq_only, without_bw_distal=without_bw_distal)
    return dataset



class CombinedDatasetNP(Dataset):
    """Combine local data and distal into Dataset, using NumPy funcions"""
    def __init__(self, data, 
                 seq_cols, 
                 cat_cols, 
                 output_col,
                 ref_genome, 
                 bed_regions, 
                 central_radius, 
                 distal_radius,
                 n_channels, 
                 bw_files, 
                 seq_only, 
                 without_bw_distal,
                 segment_task=None):
        """  
        Args:
            data: DataFrame containing local seq data and categorical data
            seq_cols: names of local seq columns
            cat_cols: names of categorical columns used for training
            output_col: name of the label column
            n_channels: number of columns (channels) in distal data to be extracted
        """
        # check input
        if not isinstance(bed_regions, BedTool):
            print(f"Error: bed_regions should be  <Bedtools>, but input is {bed_regions.__class__}!")
            sys.exit()

        if not isinstance(ref_genome, dict):
            print(f"Error : ref_genome should be <dict>, but input is {ref_genome.__class__}!")
            sys.exit()

        # Store the local seq data and label for later use
        self.data_local = data[seq_cols+[output_col]]
        
        # Sample size
        #self.n = data.shape[0]# sample size
        self.n = data.index[-1][0] + 1# batch number
        # Output column
        if output_col:
            self.y = data[output_col].astype(np.float32)
            #self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            sys.exit(f"Error: {output_col}")
            #self.y = np.zeros((self.n, 1))
        
        # Names of categorical columns
        self.cat_cols = cat_cols
        
        # Set biggest dimension for each categorical column
        self.cat_dims = [np.max(data[col]) + 1 for col in cat_cols]
        
        # Find the continuous columns
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + seq_cols + [output_col]]
        
        # Assign the continuous data to cont_X
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32)
        else:
            self.cont_X = np.zeros((self.n, 1)) 
        
        # Assign the categorical data to cat_X
        if len(self.cat_cols) > 0:
            self.cat_X = data[cat_cols]
            #self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            print("Error: no categorical data, something is wrong!", file=sys.stderr)
            sys.exit()
        
        # For distal data
        #self.h5f_path = h5f_path
        self.distal_X = None
        self.n_channels = n_channels
        self.bw_files = bw_files
        
        self.without_bw_distal = without_bw_distal
        ####
        self.bw_fh = []
        for file in self.bw_files:
            self.bw_fh.append(pyBigWig.open(file))
        ####
        self.seq_only = seq_only
        print('Number of channels to be used for distal data:', self.n_channels)
        
        self.distal_radius = distal_radius
        #self.seq_len = distal_radius*2+ central_radius - (distal_order-1) 

        self.central_radius = central_radius
        self.bed_regions = bed_regions
        self.records = ref_genome

        self.distal_info = False
        self.segment_task = segment_task
    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, index):
        """ Generate one batch of data. """
        assert index < self.n
        seqs = self.seqs_list[index]
        batch_distal = distal_encoding_by_region(seqs, self.batch_shape[index], self.distal_radius,self.records)
        if self.bw_fh:
            if not self.without_bw_distal:
                batch_annot_encoding = annot_encoding_by_region(self.bw_fh, seqs, self.batch_shape[index], self.distal_radius, self.records)
                batch_distal = np.concatenate([batch_distal, batch_annot_encoding], axis=1)
            return self.y.loc[index].values.reshape(-1, 1), self.cont_X.loc[index].values, self.cat_X.loc[index].values, batch_distal
        
       # return self.y.loc[index].values.reshape(-1, 1), self.cont_X[index], self.cat_X.loc[index].values, batch_distal
        if self.segment_task is not None:
            return self.y.loc[index].values.reshape(-1, 1), self.cat_X.loc[index].values, batch_distal
        
        return self.y.loc[index].values.reshape(-1, 1), self.cat_X.loc[index].values, batch_distal

    def get_distal_encoding_infomation(self):
        self.seqs_list,self.batch_shape = get_distal_seqs_by_region(self.bed_regions, self.records, self.distal_radius, self.central_radius)
        self.distal_info = True
        
    def get_labels(self): 
        return np.squeeze(self.y)
    
    def _get_labels(self, dataset, idx):
        return dataset.__getitem__(idx)[1]
    
def get_distal_seqs_by_region(bed_regions, seq_records, radius, segment_center):
    seqs_list = []
    batch_shape = []
    bed_generator = bed_reader(bed_regions, segment_center)
    init = False
    for batch,stand in bed_generator:
        if not init:
            chrom = batch[0].chrom
            long_seq = str(seq_records[chrom].seq)
            init = True
        else:
            if chrom != batch[0].chrom:
                chrom = batch[0].chrom
                long_seq = str(seq_records[chrom].seq)

        # Create an array to store batch after ohe encoding
       # batch_ohe_encoding = np.empty((len(batch),4,2*radius + 1), dtype='float32')
                       
        seqs = get_seqs_to_digitalized(long_seq, batch, radius, stand)
        seqs_list.append([i for i in seqs])
        batch_shape.append(len(batch))
        #digit_seqs = get_encoding_seqs(long_seq, seqs, radius, encoding, local_order, batch_ohe_encoding)

    return seqs_list,batch_shape

def get_distal_seqs_by_segments(segments, seq_records, radius):
    seqs_list = []
    batch_shape = []
    init = False
    for batch,stand in segments:
        if not init:
            chrom = batch[0].chrom
            long_seq = str(seq_records[chrom].seq)
            init = True
        else:
            if chrom != batch[0].chrom:
                chrom = batch[0].chrom
                long_seq = str(seq_records[chrom].seq)

        # Create an array to store batch after ohe encoding
       # batch_ohe_encoding = np.empty((len(batch),4,2*radius + 1), dtype='float32')
                       
        seqs = get_seqs_to_digitalized(long_seq, batch, radius, stand)
        seqs_list.append([i for i in seqs])
        batch_shape.append(len(batch))
        #digit_seqs = get_encoding_seqs(long_seq, seqs, radius, encoding, local_order, batch_ohe_encoding)

    return seqs_list,batch_shape

def seq_bpe_encoder(long_seq, start, stop, strand, radius, indices, end=False):
    """
    Encodes a sequence segment with optional imputation and strand handling.

    Args:
        long_seq (str): Full sequence.
        start (int): Start position.
        stop (int): Stop position.
        strand (str): Strand ('+' or '-').
        radius (int): Radius for window size.
        indices (list): List of indices for slicing.
        end (bool): Whether to handle end imputation.

    Returns:
        list: List of encoded sequences for each index.
    """
    # Initialize imputed sequence segments
    left_impute = right_impute = ''

    if start < 0:
        left_impute = 'N' * abs(start)
        start = 0

    if end and stop > len(long_seq):
        right_impute = 'N' * (stop - len(long_seq))

    short_seq = left_impute + long_seq[start:stop].upper() + right_impute

    # Generate encoded sequences
    window_size = 2 * radius + 1

    if strand == '+':
        return [short_seq[idx:idx + window_size] for idx in indices]
    else:
        reversed_seq = short_seq[::-1]
        return [reversed_seq[-idx - window_size: -idx] if idx else reversed_seq[-window_size:] for idx in indices]

def bpe_encoding_by_region(seqs, batch_shape, radius, seq_records):
    """
    Processes a batch of sequence regions and encodes them using seq_bpe_encoder.

    Args:
        seqs (iterable): Iterable of tuples (start, stop, chrom, strand, indices, end).
        batch_sahpe : adapte api
        radius (int): Radius for window size.
        seq_records (dict): Dictionary of sequence records.

    Returns:
        list: List of encoded sequences.
    """
    batch_bpe_encoding = []
    current_chrom = None
    long_seq = None

    for start, stop, chrom, strand, indices, end in seqs:
        # Load the sequence for a new chromosome
        if chrom != current_chrom:
            long_seq = str(seq_records[chrom].seq)
            current_chrom = chrom

        sub_batch = seq_bpe_encoder(
            long_seq=long_seq,
            start=start,
            stop=stop,
            strand=strand,
            radius=radius,
            indices=indices,
            end=end
        )
        batch_bpe_encoding.extend(sub_batch)

    return batch_bpe_encoding

def distal_encoding_by_region(seqs, batch_shape, radius,seq_records):
    #if '__iter__' not in dir(seqs):
     #   sys.exit("Error : input seqs is not <generator>!")
        
    # Create an array to store batch after ohe encoding
    batch_ohe_encoding = np.empty((batch_shape,4,2*radius + 1), dtype='float32')
    batch_index = 0
    init = True
    for start0, stop0, chrom, strand, index, end in seqs:
        if init:
            init = False
            long_seq = str(seq_records[chrom].seq)
            c = chrom
        assert chrom == c
        sub_batch_num = len(index)
        sub_batch = seq_ohe_encoder(long_seq, start0, stop0, chrom, strand, radius, index, end)

        batch_ohe_encoding[batch_index:batch_index+sub_batch_num] = sub_batch
        batch_index += sub_batch_num
    
    return batch_ohe_encoding

def kmer_encoding_by_region(seqs, batch_shape, radius,seq_records, order=3):
    #if '__iter__' not in dir(seqs):
     #   sys.exit("Error : input seqs is not <generator>!")
        
    # Create an array to store batch after ohe encoding
    batch_ohe_encoding = np.empty((batch_shape,2*radius + 1 - order + 1), dtype='float32')
    batch_index = 0
    init = True
    for start0, stop0, chrom, strand, index, end in seqs:
        if init:
            init = False
            long_seq = str(seq_records[chrom].seq)
            c = chrom
        assert chrom == c
        sub_batch_num = len(index)
        sub_batch = kmer_enc(long_seq, start0, stop0, chrom, strand, radius, index, order=order, end=end)

        batch_ohe_encoding[batch_index:batch_index+sub_batch_num] = sub_batch
        batch_index += sub_batch_num
    
    return batch_ohe_encoding

def seq_ohe_encoder(long_seq, start, stop, chrom, strand, radius, index, end=False):

    one_hot_encoder = {'A':np.array([[1,0,0,0]], dtype=np.float32).T,
               'C':np.array([[0,1,0,0]], dtype=np.float32).T,
               'G':np.array([[0,0,1,0]], dtype=np.float32).T,
               'T':np.array([[0,0,0,1]], dtype=np.float32).T,
               'R':np.array([[0.5,0,0.5,0]], dtype=np.float32).T, #A,G
               'Y':np.array([[0,0.5,0,0.5]], dtype=np.float32).T, #C,T
               'M':np.array([[0.5,0.5,0,0]], dtype=np.float32).T, #A,C
               'S':np.array([[0,0.5,0.5,0]], dtype=np.float32).T, #C,G
               'W':np.array([[0.5,0,0,0.5]], dtype=np.float32).T, #A,T
               'K':np.array([[0,0,0.5,0.5]], dtype=np.float32).T, #G,T
               'B':np.array([[0,1/3,1/3,1/3]], dtype=np.float32).T, #not A
               'D':np.array([[1/3,0,1/3,1/3]], dtype=np.float32).T, #not C
               'H':np.array([[1/3,1/3,0,1/3]], dtype=np.float32).T, #not G
               'V':np.array([[1/3,1/3,1/3,0]], dtype=np.float32).T, #not T
               'N':np.array([[0.25,0.25,0.25,0.25]], dtype=np.float32).T}

    one_hot_encoder_rc = {'A':np.array([[0,0,0,1]], dtype=np.float32).T,
               'C':np.array([[0,0,1,0]], dtype=np.float32).T,
               'G':np.array([[0,1,0,0]], dtype=np.float32).T,
               'T':np.array([[1,0,0,0]], dtype=np.float32).T,
               'R':np.array([[0,0.5,0,0.5]], dtype=np.float32).T, #A,G
               'Y':np.array([[0.5,0,0.5,0]], dtype=np.float32).T, #C,T
               'M':np.array([[0,0,0.5,0.5]], dtype=np.float32).T, #A,C
               'S':np.array([[0,0.5,0.5,0]], dtype=np.float32).T, #C,G
               'W':np.array([[0.5,0,0,0.5]], dtype=np.float32).T, #A,T
               'K':np.array([[0.5,0.5,0,0]], dtype=np.float32).T, #G,T
               'B':np.array([[1/3,1/3,1/3,0]], dtype=np.float32).T, #not A
               'D':np.array([[1/3,1/3,0,1/3]], dtype=np.float32).T, #not C
               'H':np.array([[1/3,0,1/3,1/3]], dtype=np.float32).T, #not G
               'V':np.array([[0,1/3,1/3,1/3]], dtype=np.float32).T, #not T
               'N':np.array([[0.25,0.25,0.25,0.25]], dtype=np.float32).T}
        
    #imput 
    short_seq = ['', '','']
    if start < 0:
        left_impute = 0 - start 
        start = 0
        short_seq[0] = left_impute * 'N'
        
    if end:
        long_seq_len = len(long_seq)
        right_impute = stop - long_seq_len
        short_seq[2] = right_impute * 'N'

    short_seq[1] = long_seq[start:stop].upper()

    short_seq = ''.join(short_seq)
   
   # return short_seq
    window_size = 2 * radius + 1
    if strand == '+':
        distal_seq = np.concatenate([one_hot_encoder[c] for c in short_seq], axis=1)
        distal_seq = [distal_seq[:,start1:start1 + window_size] for start1 in index]
        #distal_seq = np.expand_dims(distal_seq, 0)
    else:
        distal_seq = np.concatenate([one_hot_encoder_rc[c] for c in short_seq[::-1]], axis=1)
        distal_seq = [distal_seq[:,-start1-window_size:-start1] if start1 else distal_seq[:,-start1-window_size:] for start1 in index]    
        #distal_seq = np.expand_dims(distal_seq, 0)
    return distal_seq

#########################################################################
#                          Construct Dataset With HDF5 
# 
# Note: When sample redundancy is low, computation in preprocessing become 
#       a bottleneck im model training. This method used HD5 file to save 
#       non-redundancy distal encoding, enabling the reuse of encoding. 
#
# Suggestion: For human data, if the distal region is less than 4k,
#             it is recommended to use this method.
# 
# Dependency: h5py==3.10.0; h5py==2.10.0 can not run in multi process in this code.
#########################################################################
def prepare_dataset_h5(bed_regions, ref_genome, bw_paths, bw_files, bw_names, bw_radii,central_radius, local_radius=5, local_order=1, distal_radius=50, distal_order=1, h5f_path=None, chunk_size=5000, seq_only=True, n_h5_files=1, without_bw_distal=True):
    """Prepare the datasets for given regions, using H5 file"""
 
    # get h5f_path 
    bed_file = bed_regions.fn
    if not h5f_path:
        h5f_path = get_h5f_path(bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal)
    else:
        h5f_path = change_h5f_path(h5f_path, bed_file, bw_names, central_radius, distal_radius, distal_order, without_bw_distal)
    # Generate H5 file for distal data
    bed_file = bed_regions.fn
    process = generate_h5fv2(bed_regions, h5f_path, ref_genome, central_radius, distal_radius, distal_order, bw_paths, bw_files, chunk_size, n_h5_files, without_bw_distal)
    if process:
        process.start()
    
    # Prepare local data
    with open(ref_genome, 'r') as f:
        ref_genome = SeqIO.to_dict(SeqIO.parse(f, 'fasta'))
    start_time = time.time()
    data_local, seq_cols, categorical_features, output_feature = prepare_local_data(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only)
    print(f"local preprocess used time: {time.time() -start_time}")
    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    if process:
        process.join()
    
    # Combine local data and distal into Dataset objects
    dataset = CombinedDatasetH5(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, h5f_path=h5f_path, distal_radius=distal_radius, n_channels=n_channels)
    
    #return dataset, data_local, categorical_features
    return dataset

class CombinedDatasetH5(Dataset):
    """Combine local data and distal into Dataset, with H5"""
    def __init__(self, data, seq_cols, cat_cols, output_col, h5f_path, distal_radius, n_channels):
        """  
        Args:
            data: DataFrame containing local seq data and categorical data
            seq_cols: names of local seq columns
            cat_cols: names of categorical columns used for training
            output_col: name of the label column
            h5f_path: H5 file storing the distal data
            n_channels: number of columns (channels) in distal data to be extracted
        """
        # Store the local seq data and label for later use
        self.data_local = data[seq_cols+[output_col]]
        
        self.n = data.index[-1][0] + 1
        
        # Output column
        if output_col:
            self.y = data[output_col].astype(np.float32)
        else:
            sys.exit(f"Error: {output_col}")
        
        # Names of categorical columns
        self.cat_cols = cat_cols
        
        # Set biggest dimension for each categorical column
        self.cat_dims = [np.max(data[col]) + 1 for col in cat_cols]
        
        # Find the continuous columns
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + seq_cols + [output_col]]
        
        # Assign the continuous data to cont_X
        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1)) 
        
        # Assign the categorical data to cat_X
        if len(self.cat_cols) > 0:
            self.cat_X = data[cat_cols]
        else:
            print("Error: no categorical data, something is wrong!", file=sys.stderr)
            sys.exit()
        
        # For distal data
        self.h5f = h5py.File(h5f_path, 'r', swmr=True)
        self.distal_radius = distal_radius

    def __len__(self):
        """ Denote the total number of samples. """
        return self.n

    def __getitem__(self, index):
        """ Generate one sample of data. """
        y_values = self.y.loc[index].values.reshape(-1, 1)
        cont_X = self.cont_X[index]
        cat_X_values = self.cat_X.loc[index].values
        distal_encoding = self._read_distal(index)
        
        return y_values, cont_X, cat_X_values, distal_encoding

    def _read_distal(self, index):
        segment_encoding = self.h5f[f"segment_{index}"] # get segment
        stand = segment_encoding.attrs['stand']
        
        batch_ohe_encoding = []
        for sample_num in range(len(segment_encoding)):
            sample_dset = segment_encoding[f"sample_{sample_num}"]
            distal_seq = sample_dset[:]
            sub_index = sample_dset.attrs['index']
            sub_batch = get_sample_from_segment(distal_seq, sub_index, stand,self.distal_radius)
            batch_ohe_encoding.append(sub_batch)
        
        batch_ohe_encoding = np.concatenate(batch_ohe_encoding)

        return batch_ohe_encoding
    
def get_sample_from_segment(distal_seq, sub_index, stand,radius):
    window_size = 2 * radius + 1
    if stand == '+':
        distal_seq = [distal_seq[:,start1:start1 + window_size] for start1 in sub_index]
        #distal_seq = np.expand_dims(distal_seq, 0)
    else:
        distal_seq = [distal_seq[:,-start1-window_size:-start1] if start1 else distal_seq[:,-start1-window_size:] for start1 in sub_index]    
    return distal_seq

#########################################################################
#                          Construct DataLoader 
#########################################################################
class SiteShuffleBuffer:
    """Collect encoding windows and yield shuffled site-level training batches.

    Replaces ``generate_data_batches``, ``get_seg_share_dataset``,
    ``Create_DatasetSegment`` and the inner ``DataLoader(num_workers=0)``
    with a single streaming iterator.

    Work mode: sliding window with carry-over.

    1. Pull encoding windows from *window_iter* one at a time.
    2. Flatten each window's sites into per-feature tensors in an internal buffer.
    3. When the buffer reaches *shuffle_buffer_size*, optionally shuffle,
       then slice into *site_batch_size* batches and yield them.
    4. Leftover sites (< site_batch_size) stay in the buffer (carry-over) and
       mix with the next set of windows.
    5. At epoch end, flush all remaining sites.

    Parameters
    ----------
    window_iter : iterable
        Upstream encoding window loader (e.g. ``DataLoader(batch_size=1, collate_fn=unwrap_batch)``).
        Each item must be a dict ``{feature_name: tensor}``.
    site_batch_size : int
        Number of sites per training batch (default 256).
    shuffle_buffer_size : int
        Maximum number of sites to buffer before flushing (default 10000).
    shuffle_sites : bool
        Whether to shuffle sites within the buffer before batching.
    drop_last : bool
        Whether to drop the final incomplete batch at epoch end.
    feature_spec : FeatureBatchSpec or None
        Defines the feature order in the output tuple.
    """

    def __init__(self, window_iter, site_batch_size=256, shuffle_buffer_size=10000,
                 shuffle_sites=True, drop_last=False, feature_spec=None):
        self.window_iter = window_iter
        self.site_batch_size = site_batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_sites = shuffle_sites
        self.drop_last = drop_last
        self.feature_spec = feature_spec

        self._buffer = None     # dict of per-feature tensors
        self._n_buffered = 0

    # ------------------------------------------------------------------
    def _reset_buffer(self):
        self._buffer = {}
        self._n_buffered = 0

    def _append_window(self, window):
        """Append a single encoding window (dict of tensors) to the buffer."""
        for key, tensor in window.items():
            t = tensor.detach().cpu()
            if key not in self._buffer:
                self._buffer[key] = t.clone()
            else:
                self._buffer[key] = torch.cat([self._buffer[key], t], dim=0)
        self._n_buffered = self._buffer[list(self._buffer.keys())[0]].shape[0]

    def _feature_keys(self):
        """Return the feature keys present in the buffer, in spec order."""
        all_keys = self.feature_spec.get_feature_order() if self.feature_spec else list(self._buffer.keys())
        return [k for k in all_keys if k in self._buffer]

    def _flush(self, force_last=False):
        if self._n_buffered == 0:
            return

        # --- epoch-end: dump complete batches then optionally the remainder ---
        if force_last:
            yield from self._flush_complete()
            if self._n_buffered > 0 and not self.drop_last:
                keys = self._feature_keys()
                yield tuple(self._buffer[k] for k in keys)
            self._reset_buffer()
            return

        # --- normal mid-epoch flush ---
        yield from self._flush_complete()

    def _flush_complete(self):
        """Yield as many complete batches as possible from the buffer.

        Remaining sites (< site_batch_size) stay in the buffer (carry-over).
        """
        idx = torch.randperm(self._n_buffered) if self.shuffle_sites else torch.arange(self._n_buffered)
        n_complete = (self._n_buffered // self.site_batch_size) * self.site_batch_size

        if n_complete == 0:
            return

        batch_indices = idx[:n_complete].view(-1, self.site_batch_size)
        keys = self._feature_keys()
        for batch_idx in batch_indices:
            yield tuple(self._buffer[k][batch_idx] for k in keys)

        # carry-over
        remaining_idx = idx[n_complete:]
        if len(remaining_idx) > 0:
            for k in self._buffer:
                self._buffer[k] = self._buffer[k][remaining_idx]
            self._n_buffered = len(remaining_idx)
        else:
            self._reset_buffer()

    # ------------------------------------------------------------------
    def __iter__(self):
        self._reset_buffer()
        return self._generate()

    def _generate(self):
        for window in self.window_iter:
            self._append_window(window)
            if self._n_buffered >= self.shuffle_buffer_size:
                yield from self._flush()
        yield from self._flush(force_last=True)


def generate_data_batches(segmentLoader_train, batch_segment, batch_size, shuffle=True, sample_workers=0, use_segment_task=False):
    iter_seg_share_dataset = get_seg_share_dataset(segmentLoader_train, batch_segment, use_segment_task)
    # init
    seg_dataset = next(iter_seg_share_dataset)
    
    drop_last = False
    # gene batch to train
    while True:
        merge = False
        dataloader = DataLoader(seg_dataset, batch_size, shuffle=shuffle, num_workers=sample_workers, pin_memory=False)
        #for y, cont_x, cat_x, distal_x in dataloader:
        for batch in dataloader:
            # if sample less than batch number, merge to next segment
            if batch[0].shape[0] < batch_size:
                merge = True
                break

            yield batch
        # check end and read next segment
        try:
            seg_dataset = next(iter_seg_share_dataset)
        except StopIteration:
            # merge=True, indicate last batch not output. 
            if merge and not drop_last:
                yield batch
            return
                
        # if merge, merge to next segment
        if merge:
            seg_dataset.merge(batch)

def get_seg_share_dataset(segmentLoader, batch_segment, use_segment_task):
    if use_segment_task:
        DatasetLoader = Create_DatasetSegment_Adaptive
    else:
        DatasetLoader = Create_DatasetSegment

    count = 0
    segment_saver = []
    for segment in segmentLoader:
        segment_saver.append(segment)
        count += 1
    
        if count >= batch_segment:
            segment_dataset = DatasetLoader(segment_saver)
            yield segment_dataset
            count = 0
            segment_saver = []

    if segment_saver:
        segment_dataset = DatasetLoader(segment_saver)
        yield segment_dataset


# class Create_DatasetSegment(Dataset):
#     """     """
#     def __init__(self, data_batch):
#         """  
#         Args:
          
#         """
        
#         self.y = torch.cat([batch[0].squeeze(0) for batch in data_batch])
#         self.cat_X = torch.cat([batch[2].squeeze(0) for batch in data_batch])
#         self.distal_x = torch.cat([batch[3].squeeze(0) for batch in data_batch])
        
#         self.n = self.y.shape[0]
#         self.cont_X = np.zeros((self.n, 1))  
#     def __len__(self):
#         """ Denote the total number of samples. """
#         return self.n

#     def __getitem__(self, index):
#         """ Generate one batch of data. """
        
#         return self.y[index], self.cont_X[0], self.cat_X[index], self.distal_x[index]
    
#     def merge(self, y, cont_x, cat_x, distal_x):
#         """ Add one batch of data"""
#         self.y = torch.cat([y, self.y])
#         self.cat_X = torch.cat([cat_x, self.cat_X])
#         self.distal_x = torch.cat([distal_x, self.distal_x])

#         self.n = self.y.shape[0]

class Create_DatasetSegment(Dataset):
    """
    Dataset class for handling segmented data batches.
    """
    
    def __init__(self, data_batch):
        """
        Initialize the dataset with a batch of data.

        Args:
            data_batch (list of tensors): List of batched data where each element
                                          contains (y, cont_X, cat_X, distal_X).
                                          If data_batch contains 4 elements, cont_X is included;
                                          otherwise, cont_X will be initialized with zeros.
        """
        if len(data_batch[0]) == 4:
            self.y, self.cont_X, self.cat_X, self.distal_x = self._unpack_batch(data_batch)
        else:
            self.y, self.cat_X, self.distal_x = self._unpack_batch(data_batch, with_cont_X=False)
            self.cont_X = torch.zeros_like(self.y)  # Initialize cont_X with zeros if not provided

        self.n = self.y.shape[0]
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.n

    def __getitem__(self, index):
        """
        Retrieve a sample by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (y, cont_X, cat_X, distal_x) for the given index.
        """
        return self.y[index], self.cont_X[index], self.cat_X[index], self.distal_x[index]
    
    def merge(self, batch):
        """
        Merge a new batch of data into the existing dataset.

        Args:
            y (tensor): Target labels for the new batch.
            cont_x (tensor): Continuous features for the new batch.
            cat_x (tensor): Categorical features for the new batch.
            distal_x (tensor): Distal features for the new batch.
        """
        y, cont_x, cat_x, distal_x = batch
        self.y = torch.cat([y, self.y])
        self.cont_X = torch.cat([cont_x, self.cont_X])
        self.cat_X = torch.cat([cat_x, self.cat_X])
        self.distal_x = torch.cat([distal_x, self.distal_x])

        self.n = self.y.shape[0]

    def _unpack_batch(self, data_batch, with_cont_X=True):
        """
        Helper function to unpack a data batch into individual components.

        Args:
            data_batch (list): List of batch tensors.
            with_cont_X (bool): Whether the batch includes continuous features.

        Returns:
            tuple: Unpacked batch elements (y, cont_X, cat_X, distal_x).
        """
        if with_cont_X:
            y = torch.cat([batch[0].squeeze(0) for batch in data_batch])
            cont_X = torch.cat([batch[1].squeeze(0) for batch in data_batch])
            cat_X = torch.cat([batch[2].squeeze(0) for batch in data_batch])
            distal_x = torch.cat([batch[3].squeeze(0) for batch in data_batch])
            return y, cont_X, cat_X, distal_x
        else:
            y = torch.cat([batch[0].squeeze(0) for batch in data_batch])
            cat_X = torch.cat([batch[1].squeeze(0) for batch in data_batch])
            distal_x = torch.cat([batch[2].squeeze(0) for batch in data_batch])
            return y, cat_X, distal_x



class Create_DatasetSegment_Adaptive(Dataset):
    """
    Dataset class for handling segmented data batches.
    """
    
    def __init__(self, data_batch):
        """
        Initialize the dataset with a batch of data.

        Args:
            data_batch (list of tensors): List of batched data where each element
                                          contains (y, *features).
                                          Features can vary in number.
        """
        self.y = self._unpack_batch(data_batch, index=0)
        self.features = [self._unpack_batch(data_batch, index=i) for i in range(1, len(data_batch[0]))]
        
        self.n = self.y.shape[0]
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.n

    def __getitem__(self, index):
        """
        Retrieve a sample by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (y, *features) for the given index.
        """
        return (self.y[index],) + tuple(feature[index] for feature in self.features)
    
    def merge(self, left_data_batches):
        """
        Merge new batches of data into the existing dataset.

        Args:
            new_data_batches: Left samples of data to merge.
        """
        for i, left_data in enumerate(left_data_batches):
            if i == 0:
                #self.y = torch.cat([self.y, new_data])
                # original data is in the front
                self.y = torch.cat([left_data, self.y])
            else:
                #self.features[i-1] = torch.cat([self.features[i-1], new_data])
                self.features[i-1] = torch.cat([left_data, self.features[i-1]])
        self.n = self.y.shape[0]

    def _unpack_batch(self, data_batch, index):
        """
        Helper function to unpack a data batch into individual components.

        Args:
            data_batch (list): List of batch tensors.
            index (int): The index of the component to unpack.

        Returns:
            tensor: Unpacked component for the given index.
        """
        ndim = data_batch[0][index].ndim
        # merge left data
        if ndim == 1:
            return torch.cat([batch[index] for batch in data_batch])
        # merge multi segment data
        return torch.cat([batch[index].squeeze(0) for batch in data_batch])


######################
# DataLoader by segment
# - yield data(batch) by segment
# - sample numbers between batchs are different 
# - sample numbers is dependend on SNP density
####################

def get_expanded_region(start, stop, radius, model_type='snv'):
    """
    Calculate expanded genomic coordinates by radius 
    
    Args:
        start: Region start position (0-based inclusive)
        stop: Region end position (0-based inclusive)
        radius: Number of bases to expand around the region
        model_type: Either 'snv'  or 'indel' 
    
    Returns:
        Tuple of (expanded_start, expanded_stop)
    
    Coordinate Expansion Diagrams:
    
    SNV Model (expand according to the sampled site):
    segment: |-----|← radius →[sampled_site]← radius →|-----|
    index:    start-radius  --[    start   ]--    start+radius+1
    
    Indel Model (expand according to the sampled gap):
    segment: |----|← radius-1 →[start [   -----   ] stop]← radius-1 →|----|
    index:   start-(radius-1)  [start [sampled_gap] stop]        stop+radius
    
    Examples:
        >>> # SNV model, sampled site starts at 100.
        >>> get_expanded_region(100, 101, 10, 'snv')
        (90, 111)  # [90,130] contains original [100,120]
        
        >>> # Indel model, sampled gap between 100 and 120.
        >>> get_expanded_region(100, 120, 10, 'indel')
        (91, 130)  # [91,130] contains original [100,120]
    """
    return extend_interval(start, stop, radius, radius, model_type=model_type)


def extend_interval(start, stop, left_radius, right_radius, model_type='snv'):
    if model_type == 'snv':
        start1 = start - left_radius
        stop1 = stop + right_radius
    # 0-base [left,right]
    if model_type == 'indel':
        start1 = start - left_radius + 1
        stop1 = stop + right_radius # start + 1 + right_radius or stop-2 + right_radius
    return start1, stop1