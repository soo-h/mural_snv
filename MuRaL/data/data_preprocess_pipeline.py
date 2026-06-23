import logging
import warnings
import os

logger = logging.getLogger('mural')
import pickle
import time
import copy
from typing import List, Tuple
from typing import Dict, Any, Callable, List, Union
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from pybedtools import BedTool

from MuRaL.data.preprocessing import prepare_dataset_h5
from MuRaL.data.segment_preprocessing import prepare_soft_label, prepare_soft_label2, prepare_soft_label3, prepare_soft_labelv2,prepare_soft_label2v2 
from MuRaL.data.dataset import CombinedDatasetNPv2
from MuRaL.data.prepare_refseq_information import compute_nuc_skew
from MuRaL.data.preprocessing import distal_encoding_by_region, annot_encoding_by_region, get_distal_seqs_by_segments , kmer_encoding_by_region, bpe_encoding_by_region, bed_reader, prepare_local_feature

def prepare_dataset_npv3(segments, ref_genome, config):

    # Prepare local data
    ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    features = FeatureFactory(segments, ref_genome, config).create_all()

    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNPv2(
        segments=segments, 
        features = features,
        features_without_train = config.get('features_without_train', ['local_seq']),
        )

    return dataset

class FeatureFactory:
    """
    consider two dimensions for feature source creation:
    1. format: (bigwig, hd5, genome(computed))
    2. compute: (lazy, eager, streaming)

    Features now used include: 
        (bw, eager), (hd5, eager), (genome, eager), (genome, lazy)

    Args:
        config: {
            'features': {
                'step_avg_strategy': {
                    'loading': 'eager'
                    'type': 'hd5',
                    'path': '/path/to/file.h5'
                    'match_strategy' : 'exact',
                },
                'step_avg_kmer_mut': {
                    'loading': 'eager'
                    'type': 'hd5',
                    'path': '/path/to/file.h5',
                    'match_strategy' : 'nearest',
                },

                'nuc_skew': {
                    'loading': 'eager'
                    'type': 'computed',
                    'compute_fn': 'nuc_skew',
                },
                'local_feature': {
                    'loading': 'eager'
                    'type': 'computed',
                    'compute_fn': 'local_feature',
                    'params': {
                        'local_radius': local_radius,
                        'local_order': local_order,
                        'names': ['local_seq', 'local_seq_encode', 'mut_type'], # merge three features beacuse of the samilar calculation process
                    },
                'distal_encoding': {
                    'loading': 'lazy'
                    'type': 'computed',
                    'compute_fn': 'distal_encoding',
                    'params': {
                        'distal_radius': distal_radius,
                        'order': distal_order,
                        'encoding_type' : 'ohe',
                    },
                },
            'features_without_train': ['local_seq']
            }
        }

        """
    def __init__(self, segments,ref_genome, config: Dict[str, Any]):

        self.config = config
        self.segments = segments
        self.ref_genome = ref_genome
        self._eager_registry = {
            'bigwig':self._create_bigwig_feature,
            'hd5': self._create_hd5_feature,
            'npz': self._create_npz_feature,
            'computed': self._create_computed_feature,
            }
        self._lazy_registry = {
            'computed' : self._create_lazy_computed_feature,
            }
    
    def create_all(self):
        features = {}
        for name, feature_config in self.config['features'].items():
            feature_config = copy.deepcopy(feature_config)
            feature = self._create_feature(name, feature_config)
            features.update(feature)
        return features
    
    def _create_feature(self, name, config):
        if config['loading'] == 'eager':
            creator = self._eager_registry[config['type']]
        elif config['loading'] == 'lazy':
            creator = self._lazy_registry[config['type']]
        else:
            raise ValueError(f"Unknown loading method '{config['loading']}' for feature '{name}'")
        return creator(name, config)

    def _create_bigwig_feature(self, name: str, config: Dict[str, Any]) :
        """
        create (eager, BigWig) feature source
        
        config: {
            'type': 'bigwig',
            'path': '/path/to/file.bw'
        }
        """
        path = config.get('path')
        if not path:
            raise ValueError(f"BigWig feature '{name}' missing 'path'")
        
        feature = load_bw_features(self.segments, path)
        
        return {name: PrecomputedFeatureSource(feature)}
    
    def _create_hd5_feature(self, name: str, config: Dict[str, Any]):
        """
        create (eager, H5) feature source
        
        config: {
            'type': 'hd5',
            'path': '/path/to/file.h5'
        }
        """
        path = config.get('path')
        if not path:
            raise ValueError(f"H5 feature '{name}' missing 'path'")

        match_strategy = config.get('match_strategy', 'exact')
        if match_strategy not in ['exact', 'nearest']:
            raise ValueError(f"Unknown match strategy '{match_strategy}' for feature '{name}'")

        feature = load_hd5_features(self.segments, path, match_strategy)
        if name == 'segment_avg_kmer_mut':
            n_mut_class = config.get('n_mut_class', 3)
            feature = [f.reshape(-1,14,n_mut_class,4) for f in feature]
        return {name: PrecomputedFeatureSource(feature)}
    
    def _create_computed_feature(self, name: str, config: Dict[str, Any]):
        """
        create (eager, computed) feature source
        
        config: {
            'type': 'computed',
            'compute_fn': 'compute_nuc_skew'
        }
        """
        compute_fn_registry = {
            'nuc_skew': compute_nuc_skew,
            'local_feature' : prepare_local_feature,
        }
        compute_fn_name = config.get('compute_fn')
        compute_fn = compute_fn_registry.get(compute_fn_name)

        params = {
            **config.get('params', {}),
            'segments' : self.segments,
            'ref_genome' : self.ref_genome,
        }

        if not compute_fn:
            raise ValueError(f"Computed feature '{name}' missing 'compute_fn'")
        
        feature = compute_fn(**params)
        if isinstance(feature, dict):
            return {
                fname: PrecomputedFeatureSource(fdata) for fname, fdata in feature.items()
            }
        else:
            return {name: PrecomputedFeatureSource(feature)}
    def _create_npz_feature(self, name: str, config: Dict[str, Any]):
        """
        Create (eager, NPZ) feature source for ARG tree features
        
        Config example:
        {
            'type': 'npz',
            'path': '/path/to/chr20_hg19.npz',
            'match_strategy': 'overlap'  # 'overlap' or 'nearest'
        }
        """
        path = config.get('path')
        if not path:
            raise ValueError(f"NPZ feature '{name}' missing 'path'")
        
        # to-extend: n_trees, 与window参数相同，用于决定树序列的最大长度
        if config.get('trees_sequences') is None:
            print("Loading NPZ features with single tree per site...")
            # check match_strategy
            match_strategy = config.get('match_strategy', 'nearest')
            if match_strategy not in ['nearest', 'overlap']:
                raise ValueError(f"Unknown match strategy '{match_strategy}' for NPZ feature '{name}'")
        
            feature = load_npz_features(
                self.segments, 
                path, 
                match_strategy, 
                window=config.get('radius', 5000),
                perfix=config.get('perfix', None))
        else:
            # ====================================================================
            # 序列模式（GRU）
            # ====================================================================
            neighbor_mode = config.get('fix_trees', False)

            if neighbor_mode:
                print("Loading NPZ features with fixed number of neighbor trees per site...")
            else:
                print("Loading NPZ features with WINDOW tree sequences per site...")
            feature = load_npz_features_sequence(
                self.segments, 
                path, 
                window=config.get('radius', 5000),
                max_trees=config.get('n_trees', 9),
                padding_value=config.get('padding_value', 0.0),
                use_neighbor_mode=neighbor_mode
            )
        
        
        return {name: PrecomputedFeatureSource(feature)}
    
    def _create_lazy_computed_feature(self, name: str, config: Dict[str, Any]) :
        """
        create (lazy, computed) feature source

        """

        lazy_registry = {
            'distal_encoding' : DistalFeatureSource,
        }
        lazy_constractor = lazy_registry.get(config.get('compute_fn'))
        params = {
            **config.get('params', {}),
            'segments' : self.segments,
            'ref_genome' : self.ref_genome,
        }
        lazy_source = lazy_constractor(**params)

        return {name: lazy_source}

class FeatureSource:
    @property
    def n_channels(self) -> int:
        raise NotImplementedError
    
    def get(self, index: int):
        raise NotImplementedError

def load_bw_features(segments, path, name=None) -> np.ndarray:
    import pyBigWig

    bw = pyBigWig.open(path)
    features = []
    for batch, strand in segments:
        segment_features = [bw.values(site.chrom, site.start, site.end) for site in batch]
        features.append(segment_features)
    bw.close()
    return features

def load_hd5_features(segments, path, mode: str = 'exact') -> np.ndarray:
    import h5py
    results = []
    cache = {}
    with h5py.File(path, 'r') as h5f:
        for batch, strand in segments:
            chrom = batch[0].chrom
            key = (chrom, strand)
            # init
            if key not in cache:
                positions = h5f[chrom][strand]['positions'][:]
                features = h5f[chrom][strand]['features']
                cache[key] = (positions, features)
            # query
            positions, features = cache[key]
            query_positions = np.array([site.start for site in batch])
            
            if mode == 'exact':
                segment_features = _match_exact(positions, features, query_positions)
            else:
                segment_features = _match_nearest(positions, features, query_positions)

            results.append(segment_features)
    return results

def _match_exact(positions, features, query_positions):
    indices = np.searchsorted(positions, query_positions)
    valid_mask = (indices < len(positions)) & (positions[indices] == query_positions)

    if not np.all(valid_mask):
        raise ValueError(f"Missing positions: {query_positions[~valid_mask]}")
    unique_indices, inverse_map = np.unique(indices, return_inverse=True)
    unique_features = features[unique_indices]

    return unique_features[inverse_map]

def _match_nearest(positions, features, query_positions):
    def _find_nearest_indices(positions, query_positions):
        insert_indices = np.searchsorted(positions, query_positions)
        insert_indices = np.clip(insert_indices, 0, len(positions) - 1)

        left_indices = np.maximum(insert_indices - 1, 0)
        right_indices = np.minimum(insert_indices, len(positions) - 1)

        dist_left = np.abs(positions[left_indices] - query_positions)
        dist_right = np.abs(positions[right_indices] - query_positions)

        nearest_indices = np.where(dist_left <= dist_right, left_indices, right_indices)
        return nearest_indices

    nearest_indices = _find_nearest_indices(positions, query_positions)
    unique_indices, inverse_map = np.unique(nearest_indices, return_inverse=True)
    unique_features = features[unique_indices]
    segment_features = unique_features[inverse_map]
    return segment_features

def load_npz_features(
    segments: List[Tuple[List, str]], 
    npz_dir: str, 
    match_strategy: str = 'nearest',
    window: int = 5000,
    perfix = None
):
    """
    从NPZ文件加载ARG树特征
    
    Args:
        segments: [(batch, strand), ...] 
                 其中 batch = [site1, site2, ...]
        batch : List[BedTool.Interval], 非空, 且所有site的chrom相同
        npz_dir: NPZ文件目录
        match_strategy: 'nearest' 或 'overlap'
        window: overlap策略的窗口大小（默认±5kb）
    
    Returns:
        all_batch_features: List[np.ndarray]
                           每个元素对应一个batch的特征 [batch_size, n_features]
    """
    npz_dir = Path(npz_dir)
    
    # 按染色体分组，避免重复加载
    chrom_cache = {}
    
    all_batch_features = []
    
    for batch, strand in segments:
        chrom = batch[0].chrom
        
        # 缓存机制：只在染色体切换时加载
        if chrom not in chrom_cache:
            if perfix is not None:
                npz_path = npz_dir / f'{perfix}{chrom}_hg38.npz'
            else:
                npz_path = npz_dir / f'hgdp_tgp_sgdp_high_cov_ancients_{chrom}_hg19.npz'
            
            if not npz_path.exists():
                raise FileNotFoundError(f"NPZ file not found: {npz_path}")
            
            data = np.load(npz_path, allow_pickle=True)
            chrom_cache[chrom] = {
                'tree_intervals': data['tree_intervals'],
                'tree_features': data['tree_features']
            }
            
            print(f"load {chrom}: {len(data['tree_intervals']):,} tree, "
                  f"{data['tree_features'].shape[1]} feature")
        
        # 当前染色体的数据
        tree_intervals = chrom_cache[chrom]['tree_intervals']
        tree_features = chrom_cache[chrom]['tree_features']
        
        # 提取所有位点位置
        positions = np.array([site.start for site in batch], dtype=np.int64)
        
        # 根据策略提取特征
        if match_strategy == 'nearest':
            batch_features = extract_nearest_features(
                positions, tree_intervals, tree_features
            )
        
        elif match_strategy == 'overlap':
            batch_features = extract_overlap_features(
                positions, tree_intervals, tree_features, window
            )
        
        else:
            raise ValueError(f"Unknown strategy: {match_strategy}")

        # batch_features should be np.ndarray
        all_batch_features.append(batch_features)
    
    return all_batch_features

def extract_nearest_features(positions, tree_intervals, tree_features):
    """
    find the nearest tree features for each position
    
    Args:
        positions: [n_sites] 位点位置
        tree_intervals: [n_trees, 2] 树区间
        tree_features: [n_trees, n_features] 树特征
    
    Returns:
        features: [n_sites, n_features]
    """
    n_sites = len(positions)
    n_features = tree_features.shape[1]
    features = np.zeros((n_sites, n_features), dtype=np.float32)
    
    for i, pos in enumerate(positions):
        # 查找包含该位点的树
        containing = np.where(
            (tree_intervals[:, 0] <= pos) & (pos < tree_intervals[:, 1])
        )[0]
        
        if len(containing) > 0:
            # 取第一个（理论上应该只有一个),当前由于hg38转hg19可能有多个
            features[i] = tree_features[containing[0]]
        else:
            # 没有包含的树，找最近的
            tree_midpoints = (tree_intervals[:, 0] + tree_intervals[:, 1]) / 2
            nearest_idx = np.argmin(np.abs(tree_midpoints - pos))
            features[i] = tree_features[nearest_idx]
    
    return features

def extract_overlap_features(positions, tree_intervals, tree_features, window):
    """
    find trees in window and aggregate features
    
    Args:
        positions: [n_sites] 位点位置
        tree_intervals: [n_trees, 2] 树区间
        tree_features: [n_trees, n_features] 树特征
        window: 窗口大小（±window）
    
    Returns:
        features: [n_sites, n_features]
    """
    n_sites = len(positions)
    n_features = tree_features.shape[1]
    features = np.zeros((n_sites, n_features), dtype=np.float32)
    
    for i, pos in enumerate(positions):
        query_start = pos - window
        query_end = pos + window
        
        # 查找重叠的树
        overlapping = find_overlapping_trees(tree_intervals, query_start, query_end)
        
        if len(overlapping) == 0:
            # 没有重叠树，使用最近的树
            tree_midpoints = (tree_intervals[:, 0] + tree_intervals[:, 1]) / 2
            nearest_idx = np.argmin(np.abs(tree_midpoints - pos))
            features[i] = tree_features[nearest_idx]
        else:
            # 加权聚合
            features[i] = aggregate_tree_features(
                tree_features[overlapping],
                tree_intervals[overlapping],
                query_start, query_end, pos
            )
    
    return features


def find_overlapping_trees(tree_intervals, query_start, query_end):
    """
    查找与查询窗口重叠的树索引
    """
    # support batch query
    # if isinstance(query_start, np.ndarray) or isinstance(query_start, list):
    #     return [
    #         np.where(
    #             (tree_intervals[:, 1] > s) &
    #             (tree_intervals[:, 0] < e)
    #             )[0]
    #             for s, e in zip(query_start, query_end)
    #             ]
    overlapping = np.where(
        (tree_intervals[:, 1] > query_start) &  # 树的右端点 > 查询左端点
        (tree_intervals[:, 0] < query_end)      # 树的左端点 < 查询右端点
    )[0]
    return overlapping


def aggregate_tree_features(features, intervals, query_start, query_end, query_pos):
    """
    aggregation method: weighted average by span of trees
    """
    n_trees = len(features)
    
    # 计算每个树与查询窗口的重叠长度
    overlap_lengths = np.zeros(n_trees, dtype=np.float32)
    for i, (start, end) in enumerate(intervals):
        overlap_start = max(start, query_start)
        overlap_end = min(end, query_end)
        overlap_lengths[i] = overlap_end - overlap_start
    
    # 归一化权重
    weights = overlap_lengths / overlap_lengths.sum()
    
    # 加权平均
    aggregated = np.sum(features * weights[:, np.newaxis], axis=0)
    
    return aggregated.astype(np.float32)

def load_npz_features_sequence(
    segments: List[Tuple[List, str]], 
    npz_dir: str,
    window: int = 5000,
    max_trees: int = 9,
    padding_value: float = 0.0,
    use_neighbor_mode: bool = False,
):
    """
    加载树序列特征（用于GRU）
    
    Returns:
        all_batch_data: List[dict]
        每个dict包含:
        - 'sequences': [batch_size, max_trees, n_features]
        - 'lengths': [batch_size]
        - 'masks': [batch_size, max_trees]
    """
    npz_dir = Path(npz_dir)
    chrom_cache = {}
    all_batch_data = []
    
    for batch, strand in segments:
        if len(batch) == 0:
            continue
        
        chrom = batch[0].chrom
        
        # 缓存机制
        if chrom not in chrom_cache:
            npz_path = npz_dir / f'hgdp_tgp_sgdp_high_cov_ancients_{chrom}_hg19.npz'
            
            if not npz_path.exists():
                print(f"⚠️  NPZ文件不存在: {npz_path}")
                n_sites = len(batch)
                all_batch_data.append({
                    'sequences': np.zeros((n_sites, max_trees, 23), dtype=np.float32),
                    'lengths': np.zeros(n_sites, dtype=np.int32),
                    'masks': np.zeros((n_sites, max_trees), dtype=bool)
                })
                continue
            
            data = np.load(npz_path, allow_pickle=True)
            chrom_cache[chrom] = {
                'tree_intervals': data['tree_intervals'],
                'tree_features': data['tree_features']
            }
            
            print(f"加载 {chrom}: {len(data['tree_intervals']):,} 树")
        
        tree_intervals = chrom_cache[chrom]['tree_intervals']
        tree_features = chrom_cache[chrom]['tree_features']
        
        # 提取位点位置
        positions = np.array([site.start for site in batch], dtype=np.int64)

        if use_neighbor_mode:
            # fix tree num
            sequences = extract_neighbor_trees(
                positions,
                tree_intervals,
                tree_features,
                n_trees=max_trees,
                max_distance=window,
            )
        else:
            # 提取树序列
            sequences = extract_trees_sequences(
                positions,
                tree_intervals,
                tree_features,
                window=window,
                max_trees=max_trees,
                )
        
        
        # all_batch_data.append({
        #     'sequences': sequences,
        #     'lengths': lengths,
        #     'masks': masks
        # })
        all_batch_data.append(sequences)
    
    return all_batch_data

def extract_neighbor_trees(positions, tree_intervals, tree_features, n_trees=9, max_distance=None):
    """
    基于固定树数量提取邻居树（不依赖窗口）
    
    逻辑：
    1. 找到包含该位点的树（中心树）
    2. 向上下游各延展 (n_trees-1)//2 棵树
    3. 中心对齐，不足则padding
    
    Args:
        positions: [n_sites] 位点位置
        tree_intervals: [n_trees_total, 2] 所有树的区间
        tree_features: [n_trees_total, n_features] 所有树的特征
        n_trees: 固定返回的树数量（建议奇数）
    
    Returns:
        sequences: [n_sites, n_trees, n_features]
    """
    n_sites = len(positions)
    n_features = tree_features.shape[1]
    n_trees_total = len(tree_intervals)
    n_half = n_trees // 2
    
    sequences = np.zeros((n_sites, n_trees, n_features), dtype=np.float32)
    
    for i, pos in enumerate(positions):
        # 1. 找到中心树（包含该位点的树）
        center_tree_idx = find_containing_tree_index(pos, tree_intervals)
        
        if center_tree_idx is None:
            # 位点不在任何树中（罕见），找最近的树
            center_tree_idx = find_nearest_tree_index(pos, tree_intervals)
        # 2. 向上游延展（带距离约束）
        upstream_indices = []
        for offset in range(1, n_half + 1):
            idx = center_tree_idx - offset
            
            if idx < 0:
                # 到达染色体起始
                break
            
            if max_distance is not None:
                # 检查距离约束
                tree_end = tree_intervals[idx, 1]  # 树的右端点
                distance = pos - tree_end  # 树到位点的距离
                
                if distance > max_distance:
                    # 超出距离限制，停止延展
                    break
            
            upstream_indices.append(idx)
        
        # 反转，保持顺序（从远到近）
        upstream_indices = list(reversed(upstream_indices))
        
        # 3. 向下游延展（带距离约束）
        downstream_indices = []
        for offset in range(1, n_half + 1):
            idx = center_tree_idx + offset
            
            if idx >= n_trees_total:
                # 到达染色体结束
                break
            
            if max_distance is not None:
                # 检查距离约束
                tree_start = tree_intervals[idx, 0]  # 树的左端点
                distance = tree_start - pos  # 位点到树的距离
                
                if distance > max_distance:
                    # 超出距离限制，停止延展
                    break
            
            downstream_indices.append(idx)
        
        # 4. 合并邻居树索引
        neighbor_indices = upstream_indices + [center_tree_idx] + downstream_indices
        n_neighbors = len(neighbor_indices)
        # 5. 计算padding（中心对齐）
        n_upstream_actual = len(upstream_indices)
        left_pad = n_half - n_upstream_actual
        
        # 6. 填充特征
        sequences[i, left_pad:left_pad+n_neighbors] = tree_features[neighbor_indices]
    
    return sequences
        
def find_containing_tree_index(position, tree_intervals):
    """
    找到包含该位点的树的索引
    
    Returns:
        int or None: 树的全局索引，如果没有则返回None
    """
    containing = np.where(
        (tree_intervals[:, 0] <= position) & (position < tree_intervals[:, 1])
    )[0]
    
    return containing[0] if len(containing) > 0 else None


def find_nearest_tree_index(position, tree_intervals):
    """
    找到距离位点最近的树的索引
    
    Returns:
        int: 最近树的全局索引
    """
    midpoints = (tree_intervals[:, 0] + tree_intervals[:, 1]) / 2
    return np.argmin(np.abs(midpoints - position))

def extract_trees_sequences(position, tree_intervals, tree_features,
                          window=5000, max_trees=9):
    """
    中心对齐方案
    
    策略：
    1. 找到site相对于树的位置（上游、内部、下游）
    2. 对称地取上下游树
    3. padding时保持中心对齐
    """
    n_features = tree_features.shape[1]
    batch_features = np.zeros((len(position), max_trees, n_features), dtype=np.float32)
    n_half = max_trees // 2  # 单侧最多保留的树数
    
    # 1. 查询window内的树
    for idx, pos in enumerate(position):

        query_start = pos - window
        query_end = pos + window
        overlapping = find_overlapping_trees(tree_intervals, query_start, query_end)
    
        if len(overlapping) == 0:
            # 没有树：全部padding
            continue
    
        # 2. 找到中心树（包含site的树，或最近的树）
        center_idx = find_center_tree_in_list(
            pos, tree_intervals[overlapping], overlapping
            )
    
        # 3. 计算上下游树的数量
        n_upstream = center_idx  # 中心树之前的树
        n_downstream = len(overlapping) - center_idx - 1  # 中心树之后的树
    
        # 4. 确定选择范围（对称裁剪）
    

        upstream_keep = min(n_upstream, n_half)
        downstream_keep = min(n_downstream, n_half)
        
        select_start = center_idx - upstream_keep
        select_end = center_idx + downstream_keep + 1
        
        left_padding = n_half - upstream_keep
    
        # 5. 构建特征数组
        features = batch_features[idx]
        selected_trees = overlapping[select_start:select_end]
    
        # 填充到中心位置
        start_pos = left_padding
        end_pos = start_pos + len(selected_trees)
        features[start_pos:end_pos] = tree_features[selected_trees]
        batch_features[idx] = features
    
        # 6. 生成mask
        mask = np.zeros(max_trees, dtype=bool)
        mask[start_pos:end_pos] = True
    
        # 7. 中心位置（在max_trees维度中的索引）
        center_position = left_padding + (center_idx - select_start)
    # return features, len(selected_trees), center_position, mask
    return batch_features


def find_center_tree_in_list(position, tree_intervals_subset, original_indices):
    """
    在overlapping的树中，找到中心树
    
    返回：在subset中的索引
    """
    # 查找包含position的树
    for i, (start, end) in enumerate(tree_intervals_subset):
        if start <= position < end:
            return i
    
    # 如果没有包含，理论上不存在不包含的情况，仅在转坐标后可能发生
    midpoints = (tree_intervals_subset[:, 0] + tree_intervals_subset[:, 1]) / 2
    return np.argmin(np.abs(midpoints - position))



class PrecomputedFeatureSource(FeatureSource):
    def __init__(self, features: np.ndarray):
        self.features = features
        # 支持 numpy 数值类型 (np.float32, np.float64, np.int32, 等)
        scalar_types = (int, float, np.integer, np.floating)
        if isinstance(features[0], scalar_types):
            value = features[0]
        else:
            value = features[0][0]
        if isinstance(value, (list, np.ndarray)):
            self._n_channels = len(value)
        elif isinstance(value, scalar_types):
            self._n_channels = 1
        else:
            raise ValueError(f"Unsupported feature value type: {type(value)}")
    
    @property
    def n_channels(self) -> int:
        return self._n_channels
    
    def get(self, index: int):
        return self.features[index]
    
    def values(self):
        return self.features
    
    def __repr__(self):
        return f"PrecomputedFeatureSource(n_segment={len(self.features)}, n_channel={self._n_channels})"

class DistalFeatureSource(FeatureSource):
    """
    To Extend: bw used distal, 共享seg_list和batch_shape逻辑，将recoder替换为bw_reader即可
    当前无distsal bw需求(2025.12.15)
    """
    def __init__(self, segments, ref_genome, distal_radius, encoding_type='ohe', order=1):

        self.distal_radius = distal_radius
        self.ref_genome = ref_genome
        self._n_channels = 4
        self.encoding_type = encoding_type
        
        self.seqs_list, self.batch_shape = get_distal_seqs_by_segments(
            segments, 
            ref_genome, 
            distal_radius, 
            )

        self.encoding_fn = {
            'ohe': distal_encoding_by_region,
            'kmer' : kmer_encoding_by_region,
            'bpe' : bpe_encoding_by_region,
        }[encoding_type]
        if encoding_type == 'bpe':
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("/public/home/songhui/project/Mural/git_repo/preprocess/GROVER/")
            self.max_length = 600
        if encoding_type == 'kmer':
            from functools import partial
            self.encoding_fn = partial(self.encoding_fn, order=order)
            self._n_channels = 4 ** order

    @property
    def n_channels(self) -> int:
        return self._n_channels
    
    def get(self, index: int):
        seqs = self.seqs_list[index]
        distal_feature = self.encoding_fn(seqs, self.batch_shape[index], self.distal_radius, self.ref_genome)
        if self.encoding_type == 'bpe':
            distal_feature = [self.tokenizer(seq, max_length=self.max_length, truncation=True, padding='max_length')['input_ids']
            for seq in distal_feature
            ]

        return distal_feature
    
    def __repr__(self):
        fn_name = getattr(self.encoding_fn, '__name__', str(self.encoding_fn))
        return f"DistalFeatureSource(fn={fn_name})"



class DatasetPreprocessor:
    def __init__(self, preprocess_config, use_h5):
        self.config = preprocess_config
        self.use_h5 = use_h5

    def preprocess_dataset(self, bed_path, ref_genome, use_segment_task=False):
        bed = self.read_bed_file(bed_path)
        bed_generator = bed_reader(bed, self.config.get('segment_center'))
        bed_regions = [(batch, stand) for batch, stand in bed_generator]

        bw_files, bw_names, bw_radii = self.get_bw_paths()

        if self.use_h5:
            return self._process_h5(bed, ref_genome, bw_files, bw_names, bw_radii, use_segment_task)
        else:
            return self._process_np(bed_regions, ref_genome)

    def _process_h5(self, bed_file, ref_genome, bw_files, bw_names, bw_radii, use_segment_task):
        # H5 specific logic
        if use_segment_task:
            logger.warning("segment_task is not supported with H5 files. Ignoring segment_task.")

        step_stime = time.time()
        chunk_size = 5000
        dataset = prepare_dataset_h5(bed_file, ref_genome, bw_files, bw_names, bw_radii, 
                                     self.config['segment_center'], self.config['local_radius'],
                                     self.config['local_order'], self.config['distal_radius'],
                                     self.config['distal_order'], h5f_path=self.config['h5f_path'],
                                     chunk_size=chunk_size, seq_only=self.config['seq_only'],
                                     n_h5_files=self.config['n_h5_files'],
                                     without_bw_distal=self.config['without_bw_distal'])
        logger.info("%s preprocess with H5 used time: %.2f", bed_file.fn, (time.time() - step_stime))
        return dataset


    def _process_np(self, bed_regions, ref_genome):
        # Non-H5 logic
        logger.info('using numpy/pandas for distal_seq ...')
        step_stime = time.time()
        dataset = prepare_dataset_npv3(bed_regions, ref_genome, config=self.config)

        #if segment_task and not prediction:
            #self._save_segment_task(segment_task, self.config['trial_dir'])

        logger.info("preprocess without H5 used time: %.2f", (time.time() - step_stime))
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
            
                logger.info("bw_radii: %s", bw_radii)
            except pd.errors.EmptyDataError:
                logger.warning('no bigWig files provided in %s', bw_paths)
        else:
            logger.info('NOTE: no bigWig files provided.')
        return bw_files, bw_names, bw_radii
    
    def read_bed_file(self, file_path):
        return BedTool(file_path)