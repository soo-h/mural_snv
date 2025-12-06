import time

import pandas as pd
from Bio import SeqIO
from MuRaL.data.preprocessing import prepare_local_datav2, prepare_dataset_h5
from MuRaL.data.segment_preprocessing import prepare_soft_label, prepare_soft_label2, prepare_soft_label3, prepare_soft_labelv2,prepare_soft_label2v2 
from MuRaL.data.dataset import CombinedDatasetNPv2
from MuRaL.data.prepare_refseq_information import prepare_single_base_info
from MuRaL.data.map_segment_feature import prepare_segment_feature

from pybedtools import BedTool
import os
import pickle

def prepare_dataset_npv3(bed_regions, ref_genome, bw_files, bw_names, bw_radii,central_radius=30000, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False, 
                         without_bw_distal=False, segment_task=False, distal_encoding=None, segment_calc_method=None, path_type=None, prediction=False, segment_info_length=None,
                         single_base_task=False, segment_length_config=None, **kwargs):
    """Prepare the datasets for given regions, without an H5 file"""
    """  
        Args:
            bed_regions: <Bedtools> 
            ref_genome:  <str> path of ref genome
    
    ||--prepare_local_datav2--(prepare_segment_feature)--(prepare_single_base_info)--CombinedDatasetNPv2--||
    """
    # Prepare local data
    ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    data_local, seq_cols, categorical_features, output_feature = prepare_local_datav2(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only)

    if segment_calc_method is not None and segment_task is False:
        sys.exit("Error: method is not None, but segment_task is False. Please set --segment_task.")

    if segment_task:
        slid_strategy = kwargs.get('slid_strategy')
        step_avg_strategy = kwargs.get('step_avg_strategy')
        segment_task = prepare_segment_feature(bed_regions, central_radius, method=segment_calc_method, path_type=path_type, slid_strategy=slid_strategy, step_avg_strategy=step_avg_strategy)

    if single_base_task:
        single_base_task_config = get_single_base_task_config(single_base_task)

        print("task config : ", single_base_task_config)
        single_base_task = prepare_single_base_info(bed_regions, central_radius, ref_genome, single_base_task_config)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNPv2(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, central_radius=central_radius, distal_radius=distal_radius, n_channels=n_channels, 
                                  bw_files=bw_files, seq_only=seq_only, without_bw_distal=without_bw_distal, 
                                  segment_task=segment_task, 
                                  distal_encoding=distal_encoding, 
                                  segment_calc_method=segment_calc_method, 
                                  single_base_info=single_base_task)
    return dataset, segment_task

def prepare_dataset_npv2(bed_regions, ref_genome, bw_files, bw_names, bw_radii,central_radius=30000, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False, 
                         without_bw_distal=False, segment_task=False, distal_encoding=None, segment_calc_method=None, path_type=None, prediction=False, segment_info_length=None,
                         single_base_task=False, segment_length_config=None):
    """Prepare the datasets for given regions, without an H5 file"""
    """  
        Args:
            bed_regions: <Bedtools> 
            ref_genome:  <str> path of ref genome
    """
    # Prepare local data
    ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    data_local, seq_cols, categorical_features, output_feature = prepare_local_datav2(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only)

    if segment_calc_method is not None and segment_task is False:
        sys.exit("Error: method is not None, but segment_task is False. Please set --segment_task.")

    if segment_task:
        if segment_length_config:
            segment_length_config = get_segment_length_config(segment_length_config)
            print("segment_task: ", segment_length_config)
            segment_task = prepare_soft_label3(bed_regions, central_radius, distal_radius, segment_length_config, ref_genome, path_type)
        elif prediction:
            if segment_calc_method == 'SegMutRateByRegion':
                segment_task = prepare_soft_label2v2(bed_regions, central_radius, segment_info_length, distal_radius, ref_genome, segment_calc_method, path_type)
            else:
                segment_task = prepare_soft_label2(bed_regions, central_radius, segment_info_length, distal_radius, ref_genome, segment_calc_method, path_type)
        else:
            if segment_calc_method == 'SegMutRateByRegion':
                segment_task = prepare_soft_labelv2(bed_regions, central_radius, distal_radius, ref_genome, segment_calc_method, path_type)
            else:
                segment_task = prepare_soft_label(bed_regions, central_radius, distal_radius, ref_genome, segment_calc_method, path_type)

    if single_base_task:
        single_base_task_config = get_single_base_task_config(single_base_task)

        print("task config : ", single_base_task_config)
        single_base_task = prepare_single_base_info(bed_regions, central_radius, ref_genome, single_base_task_config)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNPv2(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, central_radius=central_radius, distal_radius=distal_radius, n_channels=n_channels, 
                                  bw_files=bw_files, seq_only=seq_only, without_bw_distal=without_bw_distal, 
                                  segment_task=segment_task, 
                                  distal_encoding=distal_encoding, 
                                  segment_calc_method=segment_calc_method, 
                                  single_base_info=single_base_task)
    return dataset, segment_task
def prepare_dataset_npv2(bed_regions, ref_genome, bw_files, bw_names, bw_radii,central_radius=30000, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False, 
                         without_bw_distal=False, segment_task=False, distal_encoding=None, segment_calc_method=None, path_type=None, prediction=False, segment_info_length=None,
                         single_base_task=False, segment_length_config=None):
    """Prepare the datasets for given regions, without an H5 file"""
    """  
        Args:
            bed_regions: <Bedtools> 
            ref_genome:  <str> path of ref genome
    """
    # Prepare local data
    ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    data_local, seq_cols, categorical_features, output_feature = prepare_local_datav2(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only)

    if segment_calc_method is not None and segment_task is False:
        sys.exit("Error: method is not None, but segment_task is False. Please set --segment_task.")

    if segment_task:
        if segment_length_config:
            segment_length_config = get_segment_length_config(segment_length_config)
            print("segment_task: ", segment_length_config)
            segment_task = prepare_soft_label3(bed_regions, central_radius, distal_radius, segment_length_config, ref_genome, path_type)
        elif prediction:
            if segment_calc_method == 'SegMutRateByRegion':
                segment_task = prepare_soft_label2v2(bed_regions, central_radius, segment_info_length, distal_radius, ref_genome, segment_calc_method, path_type)
            else:
                segment_task = prepare_soft_label2(bed_regions, central_radius, segment_info_length, distal_radius, ref_genome, segment_calc_method, path_type)
        else:
            if segment_calc_method == 'SegMutRateByRegion':
                segment_task = prepare_soft_labelv2(bed_regions, central_radius, distal_radius, ref_genome, segment_calc_method, path_type)
            else:
                segment_task = prepare_soft_label(bed_regions, central_radius, distal_radius, ref_genome, segment_calc_method, path_type)

    if single_base_task:
        single_base_task_config = get_single_base_task_config(single_base_task)

        print("task config : ", single_base_task_config)
        single_base_task = prepare_single_base_info(bed_regions, central_radius, ref_genome, single_base_task_config)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNPv2(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, central_radius=central_radius, distal_radius=distal_radius, n_channels=n_channels, 
                                  bw_files=bw_files, seq_only=seq_only, without_bw_distal=without_bw_distal, 
                                  segment_task=segment_task, 
                                  distal_encoding=distal_encoding, 
                                  segment_calc_method=segment_calc_method, 
                                  single_base_info=single_base_task)
    return dataset, segment_task


# def prepare_dataset_npv2(bed_regions, ref_genome, bw_files, bw_names, bw_radii,central_radius=30000, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False, 
#                          without_bw_distal=False, segment_task=False, distal_encoding=None, segment_calc_method=None, path_type=None, prediction=False, segment_info_length=None):
#     """Prepare the datasets for given regions, without an H5 file"""
#     """  
#         Args:
#             bed_regions: <Bedtools> 
#             ref_genome:  <str> path of ref genome
#     """
#     # Prepare local data
#     ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
#     data_local, seq_cols, categorical_features, output_feature = prepare_local_datav2(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only)

#     if segment_calc_method is not None and segment_task is False:
#         sys.exit("Error: method is not None, but segment_task is False. Please set --segment_task.")

#     if segment_task:
#         if prediction:
#             segment_task = prepare_soft_label2(bed_regions,central_radius, segment_info_length, distal_radius, ref_genome, segment_calc_method, path_type)
#         else:
#             segment_task = prepare_soft_label(bed_regions,central_radius, distal_radius, ref_genome, segment_calc_method, path_type)

#     # If seq_only flag was set, bigWig files will be ignored
#     if seq_only or without_bw_distal:
#         n_channels = 4**distal_order
#         print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
#     else:
#         n_channels = 4**distal_order + len(bw_files)
    
#     # Combine local data and distal into Dataset objects  
#     dataset = CombinedDatasetNPv2(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, central_radius=central_radius, distal_radius=distal_radius, n_channels=n_channels, 
#                                   bw_files=bw_files, seq_only=seq_only, without_bw_distal=without_bw_distal, segment_task=segment_task, distal_encoding=distal_encoding, segment_calc_method=segment_calc_method)
#     return dataset


class DatasetPreprocessor:
    def __init__(self, preprocess_config, use_h5, printer=print):
        self.config = preprocess_config
        self.use_h5 = use_h5
        self.printer = print

    def preprocess_dataset(self, bed_path, ref_genome, use_segment_task=False, distal_encoding=None, segment_calc_method=None, path_type=None, prediction=False, single_base_task=None):
        bed = self.read_bed_file(bed_path)
        bw_files, bw_names, bw_radii = self.get_bw_paths()

        if self.use_h5:
            return self._process_h5(bed, ref_genome, bw_files, bw_names, bw_radii, use_segment_task)
        else:
            return self._process_np(bed, ref_genome, bw_files, bw_names, bw_radii, use_segment_task, distal_encoding, segment_calc_method, path_type, prediction, single_base_task)

    def _process_h5(self, bed_file, ref_genome, bw_files, bw_names, bw_radii, use_segment_task):
        # H5 specific logic
        if use_segment_task:
            self.printer("Warning: segment_task is not supported with H5 files. Ignoring segment_task.")

        step_stime = time.time()
        chunk_size = 5000
        dataset = prepare_dataset_h5(bed_file, ref_genome, bw_files, bw_names, bw_radii, 
                                     self.config['segment_center'], self.config['local_radius'],
                                     self.config['local_order'], self.config['distal_radius'],
                                     self.config['distal_order'], h5f_path=self.config['h5f_path'],
                                     chunk_size=chunk_size, seq_only=self.config['seq_only'],
                                     n_h5_files=self.config['n_h5_files'],
                                     without_bw_distal=self.config['without_bw_distal'])
        self.printer(f"{bed_file.fn} preprocess with H5 used time:", (time.time() - step_stime))
        return dataset


    def _process_np(self, bed_file, ref_genome, bw_files, bw_names, bw_radii, use_segment_task, distal_encoding, segment_calc_method, path_type, prediction, single_base_task):
        # Non-H5 logic
        self.printer('using numpy/pandas for distal_seq ...')
        step_stime = time.time()
        segment_info_length = self.config.get('segment_info_length')
        step_avg_strategy = self.config.get('step_avg_strategy')
        # dataset, segment_task = prepare_dataset_npv2(bed_file, ref_genome, bw_files, bw_names, bw_radii, \
        #                              self.config['segment_center'], self.config['local_radius'], 
        #                              self.config['local_order'], self.config['distal_radius'], 
        #                              self.config['distal_order'], seq_only=self.config['seq_only'], 
        #                              without_bw_distal=self.config['without_bw_distal'],
        #                              segment_task=use_segment_task, distal_encoding=distal_encoding,
        #                              segment_calc_method=segment_calc_method, path_type=path_type, prediction=prediction,
        #                              segment_info_length=segment_info_length,
        #                              single_base_task=single_base_task,
        #                              segment_length_config=self.config.get('segment_length_config'))

        dataset, segment_task = prepare_dataset_npv3(bed_file, ref_genome, bw_files, bw_names, bw_radii, \
                                     self.config['segment_center'], self.config['local_radius'], 
                                     self.config['local_order'], self.config['distal_radius'], 
                                     self.config['distal_order'], seq_only=self.config['seq_only'], 
                                     without_bw_distal=self.config['without_bw_distal'],
                                     segment_task=use_segment_task, distal_encoding=distal_encoding,
                                     segment_calc_method=segment_calc_method, path_type=path_type, prediction=prediction,
                                     segment_info_length=segment_info_length,
                                     single_base_task=single_base_task,
                                     segment_length_config=self.config.get('segment_length_config'),
                                     slid_strategy=self.config.get('slid_strategy'),
                                     step_avg_strategy=step_avg_strategy,
                                     )


        #if segment_task and not prediction:
            #self._save_segment_task(segment_task, self.config['trial_dir'])

        self.printer(f"{bed_file.fn} preprocess without H5 used time:", (time.time() - step_stime))
        return dataset   
    
    def _save_segment_task(self, segment_task, trial_dir):
        out_name = os.path.join(trial_dir, f"segment_task.pkl")
        with open(out_name, 'wb') as pickle_file:
            pickle.dump(segment_task, pickle_file)

    def get_bw_paths(self):
        bw_files, bw_names, bw_radii = [], [], []
        bw_paths = self.config['bw_paths']
        if bw_paths:
            try:
                bw_list = pd.read_table(bw_paths, sep='\s+', header=None, comment='#')
                bw_files = list(bw_list[0])
                bw_names = list(bw_list[1])
                if bw_list.shape[1]>2:
                    bw_radii = list(bw_list[2].astype(int))
                else:
                    bw_radii = [self.config['local_radius']]*len(bw_files)
            
                self.printer("bw_radii:", bw_radii)
            except pd.errors.EmptyDataError:
                self.printer('Warnings: no bigWig files provided in', bw_paths)
        else:
            self.printer('NOTE: no bigWig files provided.')
        return bw_files, bw_names, bw_radii
    
    def read_bed_file(self, file_path):
        return BedTool(file_path)

def get_single_base_task_config(use_single_base_task):
    default_config = {
        'radius_length': 1000,
        'bin_size': 1000,
    }
    config_map = {
        'S_profile_8k_cumulated': {
            'radius_length': 8000,
            'bin_size': 1000,
            'cumulated': True,},

        'S_profile_25k_cumulated': {
            'radius_length': 25000,
            'bin_size': 1000,
            'cumulated': True,}
    }
    return config_map.get(use_single_base_task, default_config)

def get_segment_length_config(segment_length_config):
    config_map = {
        'kmer500k_avg50k': {
            'kmer_mut': 500000,
            'avg_mut': 50000,},

        'kmer300k_avg50k': {
            'kmer_mut': 300000,
            'avg_mut': 50000,},
    }
    return config_map.get(segment_length_config)