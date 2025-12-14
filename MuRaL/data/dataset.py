
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

from MuRaL.data.preprocessing import distal_encoding_by_region, annot_encoding_by_region, get_distal_seqs_by_region , kmer_encoding_by_region, bpe_encoding_by_region
from transformers import AutoTokenizer

def dict_to_tuple_collate(batch):
    """
    将 dict batch 转换为旧 Model 期望的 tuple 格式
    
    Args:
        batch: List[dict], where dict = {
            'y': tensor,
            'cat_x': tensor,
            'distal_x': tensor,
            'segment_features': tensor,  # optional
            ...
        }
    
    Returns:
        根据 batch 的键动态组装 tuple
    """
    # 1. 标准的 dict batch 化
    dict_batch = default_collate(batch)
    
    # 2. 按固定顺序转为 tuple（兼容旧 Model）
    required_keys = ['y', 'cat_x', 'distal_x']
    result = [dict_batch[key] for key in required_keys]
    
    # 3. 可选字段
    optional_keys = ['step_avg_mut', 'segment_avg_kmer_mut', 'nuc_skew']
    for key in optional_keys:
        if key in dict_batch:
            result.append(dict_batch[key])
    
    return tuple(result)

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
    def return_segment_samples(self, y, cat_x, cont_x, batch_distal, segment_label, single_base_info, index):
        return y.loc[index].values.reshape(-1, 1), cat_x.loc[index].values, batch_distal
    def return_dataset(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        samples_segment = self.return_segment_samples(y, cat_x, cont_x, batch_distal, segment_label, index)
        return OriginalMultiSegmentDataset(samples_segment)

# use annotation information in local modle
class UseAnnotInfoReturnStrategy(BaseReturnStrategy):
    def return_segment_samples(self, y, cat_x, cont_x, batch_distal, segment_label, single_base_info, index):
        return y.loc[index].values.reshape(-1, 1), cont_x.loc[index].values, cat_x.loc[index].values, batch_distal
    def return_dataset(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        samples_segment = self.return_segment_samples(y, cat_x, cont_x, batch_distal, segment_label, index)
        return UseAnnotInfoMultiSegmentDataset(samples_segment)

#  segment_task 
class SegmentTaskReturnStrategy(BaseReturnStrategy):
    def return_segment_samples(self, y, cat_x, cont_x, batch_distal, segment_label, single_base_info, index):
        sample_number = y.loc[index].values.shape[0]
        segment_id = np.tile(segment_label.get('segment_id')[index], (sample_number, 1))
        segment_id = self.conver_to_float32(segment_id)
        segment_avg_mut = np.tile(segment_label.get('segment_avg_mut')[index], (sample_number, 1))
        segment_avg_mut = self.conver_to_float32(segment_avg_mut)
        return y.loc[index].values.reshape(-1, 1), cat_x.loc[index].values, batch_distal, segment_id, segment_avg_mut

    def return_dataset(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        samples_segment = self.return_segment_samples(y, cat_x, cont_x, batch_distal, segment_label, index)
        return SegmentTaskMultiSegmentDataset(samples_segment)
    
    def conver_to_float32(self, data):
        return np.asarray(data, dtype=np.float32)
    
class SegmentMutFreqReturnStrategy(BaseReturnStrategy):
    def return_segment_samples(self, y, cat_x, cont_x, batch_distal, segment_label, single_base_info, index):
        sample_number = y.loc[index].values.shape[0]
        if len(segment_label.get('segment_avg_mut')[index]) == sample_number:
            segment_avg_mut = segment_label.get('segment_avg_mut')[index]
        elif len(segment_label.get('segment_avg_mut')[index]) == 1:
            segment_avg_mut = np.tile(segment_label.get('segment_avg_mut')[index], (sample_number, 1))
        else:
            sys.exit("Error: segment_avg_mut not match!")

        segment_avg_mut = self.conver_to_float32(segment_avg_mut)
        return y.loc[index].values.reshape(-1, 1), cat_x.loc[index].values, batch_distal, segment_avg_mut

    def return_dataset(self, y, cat_x, cont_x, batch_distal, segment_label, index):
        samples_segment = self.return_segment_samples(y, cat_x, cont_x, batch_distal, segment_label, index)
        return SegmentTaskMultiSegmentDataset(samples_segment)
    
    def conver_to_float32(self, data):
        return np.asarray(data, dtype=np.float32)

class SegmentAvgAndKmerReturnStrategy(BaseReturnStrategy):
    def return_segment_samples(self, y, cat_x, cont_x, batch_distal, segment_label, single_base_info, index):
        sample_number = y.loc[index].values.shape[0]
        if len(segment_label.get('segment_avg_mut')[index]) == sample_number:
            segment_avg_mut = segment_label.get('segment_avg_mut')[index]
        elif len(segment_label.get('segment_avg_mut')[index]) == 1:
            segment_avg_mut = np.tile(segment_label.get('segment_avg_mut')[index], (sample_number, 1))
        else:
            sys.exit("Error: segment_avg_mut not match!")
        segment_avg_mut = self.conver_to_float32(segment_avg_mut)
        if len(segment_label.get('segment_avg_kmer_mut')[index]) == sample_number:
            segment_avg_kmer_mut = segment_label.get('segment_avg_kmer_mut')[index]
        elif len(segment_label.get('segment_avg_kmer_mut')[index]) == 1:
            segment_avg_kmer_mut = np.tile(segment_label.get('segment_avg_kmer_mut')[index], (sample_number,1, 1))
        else:
            sys.exit("Error: segment_avg_kmer_mut not match!")
        segment_avg_kmer_mut = self.conver_to_float32(segment_avg_kmer_mut)

        return y.loc[index].values.reshape(-1, 1), cat_x.loc[index].values, batch_distal, segment_avg_mut, segment_avg_kmer_mut
    
    def conver_to_float32(self, data):
        return np.asarray(data, dtype=np.float32)

class SegmentMutFreqAndNucSkewReturnStrategy(BaseReturnStrategy):
    def return_segment_samples(self, y, cat_x, cont_x, batch_distal, segment_label,  single_base_info, index):
        sample_number = y.loc[index].values.shape[0]
        segment_avg_mut = np.tile(segment_label.get('segment_avg_mut')[index], (sample_number, 1))
        segment_avg_mut = self.conver_to_float32(segment_avg_mut)
        nuc_skew = single_base_info.get('nuc_skew')[index]
        nuc_skew = self.conver_to_float32(nuc_skew)
        return y.loc[index].values.reshape(-1, 1), cat_x.loc[index].values, batch_distal, segment_avg_mut, nuc_skew

    def conver_to_float32(self, data):
        return np.asarray(data, dtype=np.float32)

class StrategyError(Exception):
    pass

def return_strategy(segment_task, annot_infomation, segment_calc_method, single_base_info=None):
    if segment_task and annot_infomation:
        raise StrategyError("Error: annot and segment task function not designed.")
    elif segment_task and not annot_infomation:
        if segment_calc_method is None:
            return SegmentTaskReturnStrategy()
        elif segment_calc_method == 'SegMut' or segment_calc_method == 'SegMutRate' or segment_calc_method == 'SegMutRateByRegion':
            if single_base_info:
                return SegmentMutFreqAndNucSkewReturnStrategy()
            else:
                return SegmentMutFreqReturnStrategy()
        elif segment_calc_method == 'AvgSegMutAndKmerMut' or segment_calc_method == 'AvgStepMutAndKmerMut' or segment_calc_method == 'AvgStepMutAndKmerMutCominedLoss':
            return SegmentAvgAndKmerReturnStrategy()

    elif not segment_task and annot_infomation:
        return UseAnnotInfoReturnStrategy()
    else:
        return OriginalReturnStrategy()

########################################################################
# distal encoding Strategy
####################### 

# 弃用，因为annot feature和distal feature分开管理，只需在输入模型前将两者合并即可
# def distal_encoding_strategy(without_bw_distal, distal_encoding, distal_radius, records, bw_fh):
#     if without_bw_distal:
#         if distal_encoding == 'kmer':
#             return EncodingKmerStrategy(distal_radius, records)
#         elif distal_encoding == 'bpe':
#             return EncodingBPEStrategy(distal_radius, records)
#         return EncodingWithoutAnnotStrategy(distal_radius, records)
#     else:
#         return EncodingWithAnnotStrategy(distal_radius, records, bw_fh)

def distal_encoding_strategy(distal_encoding, distal_radius, records):
    if distal_encoding == 'kmer':
        return EncodingKmerStrategy(distal_radius, records)
    elif distal_encoding == 'bpe':
        return EncodingBPEStrategy(distal_radius, records)
    return EncodingWithoutAnnotStrategy(distal_radius, records)

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

class EncodingKmerStrategy(BaseEncodingStrategy):
    def __init__(self, distal_radius, records, bw_fh=None) -> None:
        super().__init__(distal_radius, records, bw_fh)
    def calculate(self, seqs, batch_shape):
        batch_distal = kmer_encoding_by_region(seqs, batch_shape, self.distal_radius, self.records)
        return batch_distal

class EncodingBPEStrategy(BaseEncodingStrategy):
    def __init__(self, distal_radius, records, bw_fh=None) -> None:
        super().__init__(distal_radius, records, bw_fh)
        self.tokenizer = AutoTokenizer.from_pretrained("/public/home/songhui/project/Mural/git_repo/preprocess/GROVER/")
        self.max_length = 600
    def calculate(self, seqs, batch_shape):
        batch_distal = bpe_encoding_by_region(seqs, batch_shape, self.distal_radius, self.records)
        batch_distal = [
            self.tokenizer(seq, max_length=self.max_length, truncation=True, padding='max_length')['input_ids'] 
            for seq in batch_distal
        ]
        return batch_distal
###############################
class CombinedDatasetNPv2(Dataset):
    """Combine local data and distal into Dataset, using NumPy functions"""

    def __init__(
        self, 
        data, 
        seq_cols, 
        cat_cols, 
        output_col, 
        ref_genome, 
        bed_regions, 
        central_radius, 
        distal_radius, 
        distal_order, 
        seq_only, 
        distal_encoding=None, 
        segment_calc_method=None,
        feature_sources = None,
        ):
        """
        Args:
            data: DataFrame containing local seq data and categorical data
            seq_cols: names of local seq columns
            cat_cols: names of categorical columns used for training
            output_col: name of the label column
            n_channels: number of channels (columns) in distal data to be extracted
        
        TO DO: 
            1. split distal encoding to feature source (实时生成feature)
            2. split cat_x to feature source (ComputedFeatureSource)

        """
        self._validate_inputs(bed_regions, ref_genome)
        self.seq_cols = seq_cols
        self.n = self._get_sample_size(data)
        self.y = self._get_output_labels(data, output_col)

        self.cat_cols = cat_cols
        self.cat_dims = self._calculate_cat_dims(data, cat_cols)
        self.cont_X = self._get_continuous_data(data, output_col)
        self.cat_X = self._get_categorical_data(data)
        self.distsal_X = None
        self.feature_sources = feature_sources

        self.seq_only, self.distal_radius, self.central_radius = seq_only, distal_radius, central_radius
        self.bed_regions, self.records = bed_regions, ref_genome
        self.distal_info = False

        if not use_bw:
            without_bw_distal = True

        self.distal_encoding_strategy = distal_encoding_strategy(
                                                             distal_encoding,
                                                             self.distal_radius, 
                                                             self.records, 
                                                             )

        # two question: 1. multi core 2. add function convert to tensor
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

        features ={
            'y': self.y.loc[index].values.reshape(-1, 1),
            'cat_X': self.cat_X.loc[index].values,
            'distal_X': batch_distal,
        } 
        if self.feature_sources:
            region = self.bed_regions[index]
        for feature_name in self.feature_sources:
            features[feature_name] = self.feature_sources[feature_name].get(region, index)
        return features

    def get_distal_encoding_infomation(self):
        """Get distal sequence information."""
        self.seqs_list, self.batch_shape = get_distal_seqs_by_region(self.bed_regions, self.records, self.distal_radius, self.central_radius)
        self.distal_info = True

    def get_labels(self): 
        """Return labels."""
        return np.squeeze(self.y)

    # def merge(self, dataset):
    #     """Merge datasets."""
