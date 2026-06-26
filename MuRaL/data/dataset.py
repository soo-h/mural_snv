
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


class FeatureBatchSpec:
    """Define the feature order contract for batch tuples.

    Replaces the hard-coded key lists in ``dict_to_tuple_collate``.
    Every consumer (SiteShuffleBuffer, model_train, get_inputs_labels)
    reads the same spec so the batch tuple is always consistent.
    """

    required_keys = ["mut_type", "cat_x", "distal_x"]

    optional_key_order = [
        "step_avg_mut",
        "segment_avg_kmer_mut",
        "arg_feature",
        "nuc_skew",
        "segment_id_label",  # historical strategy record, currently unused
        "sample_weight",
    ]

    def __init__(self, enabled_optional_keys=None):
        self.enabled_optional_keys = enabled_optional_keys

    def get_feature_order(self, available_keys=None):
        """Return the ordered list of feature keys for the batch tuple."""
        keys = list(self.required_keys)
        for k in self.optional_key_order:
            if self.enabled_optional_keys is not None and k not in self.enabled_optional_keys:
                continue
            if available_keys is not None and k not in available_keys:
                continue
            keys.append(k)
        return keys


def unwrap_batch(batch):
    """Collate for DataLoader with batch_size=1 — return the single item directly.

    Converts numpy arrays in dict values to torch tensors (default_collate
    would normally do this, but we bypass it).
    """
    item = batch[0]
    if isinstance(item, dict):
        return {k: torch.as_tensor(v) if isinstance(v, (np.ndarray, list)) else v
                for k, v in item.items()}
    return item


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
    optional_keys = ['step_avg_mut', 'segment_avg_kmer_mut', 'arg_feature', 'nuc_skew', 'sample_weight']
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

        if 'sample_weight' in features:
            weight = np.concatenate(features['sample_weight'].values(), axis=0).reshape(-1, 1).astype(np.float32)
            data_local = np.concatenate([data, label, weight], axis=1)
            local_radius = (data.shape[1] - 1) // 2
            col_names = (['us'+str(local_radius - i) for i in range(local_radius)] +
                        ['mid'] + ['ds'+str(i+1) for i in range(local_radius)] +
                        ['mut_type', 'sample_weight'])
        else:
            data_local = np.concatenate([data, label], axis=1)
            local_radius = (data.shape[1] - 1) // 2
            col_names = (['us'+str(local_radius - i) for i in range(local_radius)] +
                        ['mid'] + ['ds'+str(i+1) for i in range(local_radius)] +
                        ['mut_type'])

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


class EncodingWindowDataset(Dataset):
    """Map-style dataset that returns a window-level dict of features.

    Replacement for ``CombinedDatasetNPv2``.  The public API is identical:
    ``__getitem__(index)`` returns a dict with one entry per feature.
    """

    def __init__(self, segments, features, features_without_train=None):
        if features_without_train is None:
            features_without_train = ['local_seq']
        self.segments = segments
        self.features = features
        self.features_without_train = features_without_train

        # Pre-compute cat_dims for compatibility (used by model config)
        self.cat_dims = self._calculate_cat_dims(features)
        self._build_data_local(features)

    def _build_data_local(self, features):
        assert 'local_seq' in features, "Error: local_seq must be in features"
        assert 'mut_type' in features, "Error: mut_type must be in features"
        data = np.concatenate(features['local_seq'].values(), axis=0)
        label = np.concatenate(features['mut_type'].values(), axis=0).reshape(-1, 1).astype(int)

        if 'sample_weight' in features:
            weight = np.concatenate(features['sample_weight'].values(), axis=0).reshape(-1, 1).astype(np.float32)
            data_local = np.concatenate([data, label, weight], axis=1)
            local_radius = (data.shape[1] - 1) // 2
            col_names = (['us'+str(local_radius - i) for i in range(local_radius)] +
                        ['mid'] + ['ds'+str(i+1) for i in range(local_radius)] +
                        ['mut_type', 'sample_weight'])
        else:
            data_local = np.concatenate([data, label], axis=1)
            local_radius = (data.shape[1] - 1) // 2
            col_names = (['us'+str(local_radius - i) for i in range(local_radius)] +
                        ['mid'] + ['ds'+str(i+1) for i in range(local_radius)] +
                        ['mut_type'])

        self.data_local = pd.DataFrame(data_local, columns=col_names)

    def _calculate_cat_dims(self, features):
        return np.concatenate(features['cat_x'].values(), axis=0).max(axis=0) + 1

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        features = {}
        for feature_name in self.features:
            if feature_name in self.features_without_train:
                continue
            features[feature_name] = self.features[feature_name].get(index)
        return features

    def get_labels(self):
        return self.data_local['mut_type'].values
