
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
# from transformers import AutoTokenizer
from torch.utils.data.dataloader import default_collate


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
    required_keys = ['mut_type', 'cat_x', 'distal_x']
    result = [dict_batch[key] for key in required_keys]
    
    # 3. 可选字段
    optional_keys = ['step_avg_mut', 'segment_avg_kmer_mut', 'nuc_skew']
    for key in optional_keys:
        if key in dict_batch:
            result.append(dict_batch[key])
    
    return tuple(result)

###############################
class CombinedDatasetNPv2(Dataset):
    """Combine local data and distal into Dataset, using NumPy functions"""

    def __init__(
        self, 
        segments, 
        features = None,
        features_without_train = ['local_seq'],
        ):
        """
        Args:
            data: DataFrame containing local seq data and categorical data
        Note:
            features: encode sample by segment(region) to reduce IO time
                So, all features is nested structure(except y), e.g., 
                features['local_seq_encode'][i][m] is the local_seq_encode of i-th segment(region), m-th sample in this segment
        """
        self.data_local = self._build_data_local(features)
        self.cat_dims = self._calculate_cat_dims(features)

        self.features_without_train = features_without_train
        self.features = features

        self.segments = segments

    def _build_data_local(self, features):
        assert 'local_seq' in features, "Error: local_seq must be in features"
        assert 'mut_type' in features, "Error: mut_type must be in features"
        data = np.concatenate(features['local_seq'].values(), axis=0) # (segment, sample, local_seq_len) --> (sample_total, local_seq_len)
        label = np.concatenate(features['mut_type'].values(), axis=0).reshape(-1, 1).astype(int) # (segment, sample) --> (sample_total, 1)
        data_local = np.concatenate([data, label], axis=1)
        local_radius = (data.shape[1] - 1) // 2
        col_names = ['us'+str(local_radius - i) for i in range(local_radius)] + ['mid'] + ['ds'+str(i+1) for i in range(local_radius)] + ['mut_type']
        return pd.DataFrame(data_local, columns=col_names)
        
    def _calculate_cat_dims(self, features):
        """Calculate dimensions of local_seq_encode columns."""
        # max_cat = None
        # for segment in features['local_seq_encode']:
        #     if max_cat is None:
        #         max_cat = np.max(segment, axis=0)
        #     else:
        #         max_cat = np.maximum(max_cat, np.max(segment, axis=0))
        # return max_cat + 1
        return np.concatenate(features['cat_x'].values(), axis=0).max(axis=0) + 1

    def __len__(self):
        """Total number of segment."""
        return len(self.segments)

    def __getitem__(self, index):
        """Generate one batch of data."""
        features = {}
        for feature_name in self.features:
            if feature_name in self.features_without_train:
                continue
            features[feature_name] = self.features[feature_name].get(index)
        return features

    def get_labels(self): 
        """Return labels."""
        return self.data_local['mut_type'].values
