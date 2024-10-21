import time

import pandas as pd
from Bio import SeqIO
from MuRaL.data.preprocessing import prepare_local_datav2, prepare_soft_label, prepare_dataset_h5
from MuRaL.data.dataset import CombinedDatasetNPv2

from pybedtools import BedTool




def prepare_dataset_npv2(bed_regions, ref_genome, bw_files, bw_names, bw_radii,central_radius=30000, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False, without_bw_distal=False, segment_task=False):
    """Prepare the datasets for given regions, without an H5 file"""
    """  
        Args:
            bed_regions: <Bedtools> 
            ref_genome:  <str> path of ref genome
    """
    # Prepare local data
    ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    data_local, seq_cols, categorical_features, output_feature = prepare_local_datav2(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only)

    if segment_task:
        segment_task = prepare_soft_label(bed_regions,central_radius, distal_radius)

    # If seq_only flag was set, bigWig files will be ignored
    if seq_only or without_bw_distal:
        n_channels = 4**distal_order
        print('NOTE: seq_only/without_bw_distal was set, so skip bigwig tracks for distal regions!')
    else:
        n_channels = 4**distal_order + len(bw_files)
    
    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNPv2(data=data_local, seq_cols=seq_cols, cat_cols=categorical_features, output_col=output_feature, ref_genome=ref_genome, bed_regions=bed_regions, central_radius=central_radius, distal_radius=distal_radius, n_channels=n_channels, bw_files=bw_files, seq_only=seq_only, without_bw_distal=without_bw_distal, segment_task=segment_task)
    return dataset


class DatasetPreprocessor:
    def __init__(self, preprocess_config, use_h5, printer=print):
        self.config = preprocess_config
        self.use_h5 = use_h5
        self.printer = print

    def preprocess_dataset(self, bed_path, ref_genome, use_segment_task=False):
        bed = self.read_bed_file(bed_path)
        bw_files, bw_names, bw_radii = self.get_bw_paths()

        if self.use_h5:
            return self._process_h5(bed, ref_genome, bw_files, bw_names, bw_radii, use_segment_task)
        else:
            return self._process_np(bed, ref_genome, bw_files, bw_names, bw_radii, use_segment_task)

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


    def _process_np(self, bed_file, ref_genome, bw_files, bw_names, bw_radii, use_segment_task):
        # Non-H5 logic
        self.printer('using numpy/pandas for distal_seq ...')
        step_stime = time.time()
        dataset = prepare_dataset_npv2(bed_file, ref_genome, bw_files, bw_names, bw_radii, \
                                     self.config['segment_center'], self.config['local_radius'], 
                                     self.config['local_order'], self.config['distal_radius'], 
                                     self.config['distal_order'], seq_only=self.config['seq_only'], 
                                     without_bw_distal=self.config['without_bw_distal'],
                                     segment_task=use_segment_task)

        self.printer(f"{bed_file.fn} preprocess without H5 used time:", (time.time() - step_stime))
        return dataset   
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
