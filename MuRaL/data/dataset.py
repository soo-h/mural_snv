
#from janggu.data import Bioseq, Cover
import sys
import pyBigWig
from pybedtools import BedTool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from MuRaL.data.preprocessing import distal_encoding_by_region, annot_encoding_by_region, get_distal_seqs_by_region 

##########################
######## Return Strategy

class OriginalMultiSegmentDataset(Dataset):
    
    def __init__(self, samples_segment) -> None:
        super().__init__()
        try:
            self.y, self.cat_x, self.distal_x = samples_segment
        except:
            sys.exit("Error: samples_segment not match!")

        self.n = len(self.y)
    
    def _updata_n(self):
        self.n = len(self.y)

    def __len__(self):
        """Total number of samples."""
        return self.n

    def __getitem__(self, index):
        """Generate one batch of data."""
        return self.y[index], self.cat_x[index], self.distal_x[index]

    def merge_dataset(self, dataset):
        self.y = torch.cat([self.y, dataset.y])
        self.cat_x = torch.cat([self.cat_x, dataset.cat_x])
        self.distal_x = torch.cat([self.distal_x, dataset.distal_x])
        self._updata_n()
    
    def merge_batch(self, samples_segment):
        try:
            y, cat_x, distal_x = samples_segment
        except:
            sys.exit("Error: samples_segment not match in merge batch !")
        self.y = torch.cat([self.y, y])
        self.cat_x = torch.cat([self.cat_x, cat_x])
        self.distal_x = torch.cat([self.distal_x, distal_x])
        self._updata_n()

class UseAnnotInfoMultiSegmentDataset(Dataset):
    def __init__(self, samples_segment) -> None:
        super().__init__()
        try:
            self.y, self.cont_x, self.cat_x, self.distal_x = samples_segment
        except:
            sys.exit("Error: samples_segment not match!")
        
        self.n = len(self.y)
    
    def _updata_n(self):
        self.n = len(self.y)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.y[index], self.cont_x[index], self.cat_x[index], self.distal_x[index]
    
    def merge_dataset(self, dataset):
        self.y = torch.cat([self.y, dataset.y])
        self.cont_x = torch.cat([self.cont_x, dataset.cont_x])
        self.cat_x = torch.cat([self.cat_x, dataset.cat_x])
        self.distal_x = torch.cat([self.distal_x, dataset.distal_x])
        self.updata_n()
    
    def merge_batch(self, samples_segment):
        try:
            y, cont_x, cat_x, distal_x = samples_segment
        except:
            sys.exit("Error: samples_segment not match in merge batch !")
        self.y = torch.cat([self.y, y])
        self.cont_x = torch.cat([self.cont_x, cont_x])
        self.cat_x = torch.cat([self.cat_x, cat_x])
        self.distal_x = torch.cat([self.distal_x, distal_x])
        self._updata_n()
    
class SegmentTaskMultiSegmentDataset(Dataset):
    def __init__(self, samples_segment) -> None:
        super().__init__()

        try:
            self.y, self.cat_x, self.distal_x, self.segment_label = samples_segment
        except:
            sys.exit("Error: samples_segment not match!")
        
        self.n = len(self.y)
        
    def __len__(self):
        return len(self.y)
    
    def _updata_n(self):
        self.n = len(self.y)
    
    def __getitem__(self, index):
        return self.y[index], self.cat_x[index], self.distal_x[index], self.segment_label[index]
    
    def merge_dataset(self, dataset):
        self.y = torch.cat([self.y, dataset.y])
        self.cat_x = torch.cat([self.cat_x, dataset.cat_x])
        self.distal_x = torch.cat([self.distal_x, dataset.distal_x])
        self.segment_label = torch.cat([self.segment_label, dataset.segment_label])
        self._updata_n()
    
    def merge_batch(self, samples_segment):
        try:
            y, cat_x, distal_x, segment_label = samples_segment
        except:
            sys.exit("Error: samples_segment not match in merge batch !")
        self.y = torch.cat([self.y, y])
        self.cat_x = torch.cat([self.cat_x, cat_x])
        self.distal_x = torch.cat([self.distal_x, distal_x])
        self.segment_label = torch.cat([self.segment_label, segment_label])
        self._updata_n()

# 定义基类策略
class BaseReturnStrategy:
    def return_segment_samples(self, y, cat_X, cont_x, batch_distal, segment_label, index):
        raise NotImplementedError("Subclasses should implement this!")
    
    def return_dataset(self, *args):
        raise NotImplementedError("Subclasses should implement this!")

# no segment task and no annotation information
class OriginalReturnStrategy(BaseReturnStrategy):
    def return_segment_samples(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        return y.loc[index].values.reshape(-1, 1), cat_x.loc[index].values, batch_distal
    def return_dataset(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        samples_segment = self.return_segment_samples(y, cat_x, cont_x, batch_distal, segment_label, index)
        return OriginalMultiSegmentDataset(samples_segment)

# use annotation information in local modle
class UseAnnotInfoReturnStrategy(BaseReturnStrategy):
    def return_segment_samples(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        return y.loc[index].values.reshape(-1, 1), cont_x.loc[index].values, cat_x.loc[index].values, batch_distal
    def return_dataset(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        samples_segment = self.return_segment_samples(y, cat_x, cont_x, batch_distal, segment_label, index)
        return UseAnnotInfoMultiSegmentDataset(samples_segment)

#  segment_task 
class SegmentTaskReturnStrategy(BaseReturnStrategy):
    def return_segment_samples(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        segment_id = segment_label.get('segment_id')
        segment_avg_mut = segment_label.get('segment_avg_mut')
        return y.loc[index].values.reshape(-1, 1), cat_x.loc[index].values, batch_distal, segment_id[index], segment_avg_mut[index]

    def return_dataset(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        samples_segment = self.return_segment_samples(y, cat_x, cont_x, batch_distal, segment_label, index)
        return SegmentTaskMultiSegmentDataset(samples_segment)
    
class StrategyError(Exception):
    pass

def return_strategy(segment_task, annot_infomation):
    if segment_task and annot_infomation:
        raise StrategyError("Error: annot and segment task function not designed.")
    elif segment_task and not annot_infomation:
        return SegmentTaskReturnStrategy()
    elif not segment_task and annot_infomation:
        return UseAnnotInfoReturnStrategy()
    else:
        return OriginalReturnStrategy()

########################################################################
# distal encoding Strategy
####################### 
def distal_encoding_strategy(without_bw_distal, distal_radius, records, bw_fh):
    if without_bw_distal:
        return EncodingWithoutAnnotStrategy(distal_radius, records)
    else:
        return EncodingWithAnnotStrategy(distal_radius, records, bw_fh)

class BaseEncodingStrategy:
    def __init__(self, distal_radius, records, bw_fh=None) -> None:
        self.distal_radius = distal_radius
        self.records = records
        self.bw_fh = bw_fh
    def calculate(self):
        raise NotImplementedError("Subclasses should implement this!")

class EncodingWithoutAnnotStrategy(BaseEncodingStrategy):
    def __init__(self, distal_radius, records) -> None:
        super().__init__(distal_radius, records)
    
    def calculate(self, seqs, batch_shape):
        batch_distal = distal_encoding_by_region(seqs, batch_shape, self.distal_radius, self.records)
        return batch_distal
        
class EncodingWithAnnotStrategy(BaseEncodingStrategy):
    def __init__(self, distal_radius, records, bw_fh) -> None:
        super().__init__(distal_radius, records, bw_fh)
    
    def calculate(self, seqs, batch_shape):
        batch_distal = distal_encoding_by_region(seqs, batch_shape, self.distal_radius, self.records)
        batch_annot_encoding = annot_encoding_by_region(self.bw_fh, seqs, batch_shape, self.distal_radius, self.records)
        batch_distal = np.concatenate([batch_distal, batch_annot_encoding], axis=1)
        return batch_distal


###############################
class CombinedDatasetNPv2(Dataset):
    """Combine local data and distal into Dataset, using NumPy functions"""

    def __init__(self, data, seq_cols, cat_cols, output_col, 
                 ref_genome, bed_regions, central_radius, distal_radius, 
                 n_channels, bw_files, seq_only, without_bw_distal, 
                 segment_task=False):
        """
        Args:
            data: DataFrame containing local seq data and categorical data
            seq_cols: names of local seq columns
            cat_cols: names of categorical columns used for training
            output_col: name of the label column
            n_channels: number of channels (columns) in distal data to be extracted
        """
        self._validate_inputs(bed_regions, ref_genome)
        self.seq_cols = seq_cols
        self.data_local = self._process_local_data(data, seq_cols, output_col)
        self.n = self._get_sample_size(data)
        self.y = self._get_output_labels(data, output_col)

        self.cat_cols = cat_cols
        self.cat_dims = self._calculate_cat_dims(data, cat_cols)
        self.cont_X = self._get_continuous_data(data, output_col)
        self.cat_X = self._get_categorical_data(data)
        self.distsal_X = None

        if bw_files:
            use_bw = True
            self.bw_fh = self._open_bw_files(bw_files)
        else:
            use_bw = False
            self.bw_fh = []
        
        if segment_task is False:
            set_segment_task = False
        else:
            set_segment_task = True

        self.n_channels, self.seq_only, self.distal_radius, self.central_radius = n_channels, seq_only, distal_radius, central_radius
        self.bed_regions, self.records = bed_regions, ref_genome
        self.distal_info, self.segment_task = False, segment_task

        if not use_bw:
            without_bw_distal = True

        self.distal_encoding_strategy = distal_encoding_strategy(without_bw_distal, 
                                                             self.distal_radius, 
                                                             self.records, 
                                                             self.bw_fh)

        self.return_strategy = return_strategy(set_segment_task, use_bw)
        # two question: 1. multi core 2. add function convert to tensor
        #self.return_strategy = self.return_strategy.return_dataset
        self.return_strategy = self.return_strategy.return_segment_samples

        self.get_distal_encoding_infomation()

    def _validate_inputs(self, bed_regions, ref_genome):
        """Ensure inputs are correct."""
        if not isinstance(bed_regions, BedTool):
            raise TypeError(f"Error: bed_regions should be <Bedtools>, but got {type(bed_regions)}")
        if not isinstance(ref_genome, dict):
            raise TypeError(f"Error: ref_genome should be <dict>, but got {type(ref_genome)}")

    def _process_local_data(self, data, seq_cols, output_col):
        """Process local sequence data."""
        return data[seq_cols + [output_col]]

    def _get_sample_size(self, data):
        """Return the number of samples (batch size)."""
        return data.index[-1][0] + 1

    def _get_output_labels(self, data, output_col):
        """Process output labels."""
        if output_col:
            return data[output_col].astype(np.float32)
        raise ValueError(f"Error: missing output column {output_col}")

    def _calculate_cat_dims(self, data, cat_cols):
        """Calculate dimensions of categorical columns."""
        return [np.max(data[col]) + 1 for col in cat_cols]

    def _get_continuous_data(self, data, output_col):
        """Extract continuous features."""
        self.cont_cols = [col for col in data.columns if col not in self.cat_cols + self.seq_cols + [output_col]]
        return data[self.cont_cols].astype(np.float32) if self.cont_cols else np.zeros((self.n, 1))

    def _get_categorical_data(self, data):
        """Extract categorical features."""
        if self.cat_cols:
            return data[self.cat_cols]
        raise ValueError("Error: no categorical data found")

    def _open_bw_files(self, bw_files):
        """Open BigWig files."""
        return [pyBigWig.open(file) for file in bw_files]

    def __len__(self):
        """Total number of samples."""
        return self.n

    def __getitem__(self, index):
        """Generate one batch of data."""
        seqs = self.seqs_list[index]
        batch_distal = self.distal_encoding_strategy.calculate(seqs, self.batch_shape[index])
        return self.return_strategy(self.y, self.cat_X, self.cont_X, batch_distal, self.segment_task, index)

    def get_distal_encoding_infomation(self):
        """Get distal sequence information."""
        self.seqs_list, self.batch_shape = get_distal_seqs_by_region(self.bed_regions, self.records, self.distal_radius, self.central_radius)
        self.distal_info = True

    def get_labels(self): 
        """Return labels."""
        return np.squeeze(self.y)

    # def merge(self, dataset):
    #     """Merge datasets."""
