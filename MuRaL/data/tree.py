# env: py312_clean
# 总计训练特征: 4+6+4+3+6+3 = 26 个
# 元数据: 2 个

# # 方法1: 标准化 (Z-score) - 推荐
# normalized = (X - mean) / std

# # 方法2: Min-Max scaling
# normalized = (X - min) / (max - min)

# # 方法3: Robust scaling (对outliers鲁棒)
# normalized = (X - median) / IQR

"""
ARG树特征提取工具
从TreeSequence文件提取所有特征，保存为区间索引的HDF5格式
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
from scipy.stats import skew
import h5py
import tskit
import tszip
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 特征定义
# ============================================================================

TREE_FEATURE_GROUPS = {
    # === 核心特征组（用于训练）===
    
    # 1. 树的基本属性 (4)
    'tree_properties': [
        'tree_span_norm',       # 树覆盖长度（归一化）
        'n_nodes',              # 总节点数
        'n_internal_nodes',     # 内部节点数
        'n_sites',              # 分离位点数
    ],
    
    # 2. 分支长度特征 (7)
    'branch_statistics': [
        'total_branch_length',  # TBL
        'mean_branch_length',   
        'median_branch_length', 
        'std_branch_length',    
        'max_branch_length',    
        'internal_branch_length', 
        'external_branch_prop', 
    ],
    
    # 3. 合并时间特征 (4)
    'coalescent_statistics': [
        'tmrca',                
        'mean_coal_time',       
        'median_coal_time',     
        'std_coal_time',        
    ],
    
    # 4. 突变特征 (3)
    'mutation_statistics': [
        'n_mutations',          
        'mutations_per_bp',     
        'mutations_per_tbl',    
    ],
    
    # 5. 等位基因频谱 (5)
    'allele_frequency_spectrum': [
        'afs_singleton_prop',   
        'afs_doubleton_prop',   
        'afs_rare_prop',        
        'afs_low_freq_prop',    
        'afs_common_prop',      
    ],
    
    # 6. 衍生/复合特征 (3)
    'derived_features': [
        'branch_length_cv',     
        'time_skewness',        
        'diversity_theta',      
    ],
    
    # === 元数据（不用于训练）===
    'metadata': [
    ]
}

# 要删除的特征 (feature 分析表明信息冗余)
FEATURE_TO_DEL = [
    'external_branch_prop', # 与AFS_singleton完全相同
    'diversity_theta',     # 与n_sites高度相关,由于sample size是定制，导致theta是segsite的线性缩放
]

# 获取所有特征名称（按顺序）
def get_all_feature_names(include_metadata=False):
    """获取所有特征名称的有序列表"""
    features = []
    for group_name in ['tree_properties', 'branch_statistics', 
                       'coalescent_statistics', 'mutation_statistics',
                       'allele_frequency_spectrum', 'derived_features']:
        features.extend(TREE_FEATURE_GROUPS[group_name])
    
    if include_metadata:
        features.extend(TREE_FEATURE_GROUPS['metadata'])
    
    return features

ALL_TRAINING_FEATURES = get_all_feature_names(include_metadata=False)  # 25个
ALL_METADATA_FEATURES = TREE_FEATURE_GROUPS['metadata']  # 2个
TOTAL_FEATURES = len(ALL_TRAINING_FEATURES)  # 25


# ============================================================================
# 辅助函数
# ============================================================================

def compute_branch_length_stats(tree) -> Dict[str, float]:
    """
    计算分支长度的统计量
    
    返回: {mean, median, std, max, cv}
    """
    branch_lengths = []
    external_bl = 0.0
    internal_bl = 0.0
    
    for node in tree.nodes():
        # if node == tree.root:
        if node == tree.roots:
            continue
            
        bl = tree.branch_length(node)
        branch_lengths.append(bl)
        
        if tree.is_sample(node):
            external_bl += bl
        else:
            internal_bl += bl
    
    if len(branch_lengths) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'max': 0.0,
            'cv': 0.0,
            'external_bl': 0.0,
            'internal_bl': 0.0
        }
    
    mean_bl = np.mean(branch_lengths)
    std_bl = np.std(branch_lengths)
    
    return {
        'mean': mean_bl,
        'median': np.median(branch_lengths),
        'std': std_bl,
        'max': np.max(branch_lengths),
        'cv': std_bl / mean_bl if mean_bl > 0 else 0.0,
        'external_bl': external_bl,
        'internal_bl': internal_bl
    }


def compute_coalescent_time_stats(tree, ts) -> Dict[str, float]:
    """
    计算合并时间的统计量
    
    返回: {tmrca, mean, median, std, skewness}
    """
    times = []
    
    for node in tree.nodes():
        t = tree.time(node)
        if t > 0:  # 排除当代样本
            times.append(t)
    
    if len(times) == 0:
        return {
            'tmrca': 0.0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'skewness': 0.0
        }
    
    # tmrca = ts.node(tree.root).time
    root_times = [ts.node(root).time for root in tree.roots]
    tmrca = max(root_times) if root_times else 0.0
    
    return {
        'tmrca': tmrca,
        'mean': np.mean(times),
        'median': np.median(times),
        'std': np.std(times),
        'skewness': skew(times) if len(times) > 2 else 0.0
    }


def compute_afs_features(tree, n_samples: int) -> Dict[str, float]:
    """
    计算等位基因频谱特征
    
    返回: {singleton_prop, doubleton_prop, rare_prop, low_freq_prop, common_prop}
    """
    afs_bl = {}
    
    for node in tree.nodes():
        # if node == tree.root:
        if node == tree.roots:
            continue
        
        num_samples = tree.num_samples(node)
        branch_length = tree.branch_length(node)
        
        if num_samples not in afs_bl:
            afs_bl[num_samples] = 0.0
        afs_bl[num_samples] += branch_length
    
    total_bl = tree.total_branch_length
    
    if total_bl == 0:
        return {
            'singleton_prop': 0.0,
            'doubleton_prop': 0.0,
            'rare_prop': 0.0,
            'low_freq_prop': 0.0,
            'common_prop': 0.0
        }
    
    # 计算各类别的分支长度
    singleton_bl = afs_bl.get(1, 0.0)
    doubleton_bl = afs_bl.get(2, 0.0)
    
    # 稀有变异（AC ≤ 5%）
    rare_threshold = max(2, int(0.05 * n_samples))
    rare_bl = sum(afs_bl.get(ac, 0.0) for ac in range(1, rare_threshold + 1))
    
    # 低频变异（5% < AC ≤ 10%）
    low_freq_threshold = int(0.10 * n_samples)
    low_freq_bl = sum(afs_bl.get(ac, 0.0) 
                     for ac in range(rare_threshold + 1, low_freq_threshold + 1))
    
    # 常见变异（AC > 10%）
    common_bl = sum(afs_bl.get(ac, 0.0) 
                   for ac in range(low_freq_threshold + 1, n_samples))
    
    return {
        'singleton_prop': singleton_bl / total_bl,
        'doubleton_prop': doubleton_bl / total_bl,
        'rare_prop': rare_bl / total_bl,
        'low_freq_prop': low_freq_bl / total_bl,
        'common_prop': common_bl / total_bl
    }


def compute_diversity_theta(tree, n_samples: int) -> float:
    """
    计算Watterson's theta估计
    theta = S / a_n
    其中 S 是分离位点数，a_n = sum(1/i for i=1 to n-1)
    """
    n_sites = tree.num_sites
    
    if n_samples <= 1:
        return 0.0
    
    # 计算调和数 a_n
    a_n = sum(1.0 / i for i in range(1, n_samples))
    
    if a_n == 0:
        return 0.0
    
    return n_sites / a_n


# ============================================================================
# 核心特征提取函数
# ============================================================================

def extract_features_from_tree(tree, ts, genome_length: float) -> np.ndarray:
    """
    从单棵树提取所有特征
    
    参数:
        tree: tskit.Tree对象
        ts: tskit.TreeSequence对象
        genome_length: 染色体总长度
    
    返回:
        features: numpy array, shape=(TOTAL_FEATURES + 2,)
                 前TOTAL_FEATURES个是训练特征，后2个是元数据
    """
    n_samples = ts.num_samples
    interval = tree.interval
    tree_start, tree_end = interval.left, interval.right
    tree_span = tree_end - tree_start
    
    # 初始化特征字典
    feature_dict = {}
    
    # === 1. 树的基本属性 ===
    feature_dict['tree_span_norm'] = tree_span / genome_length
    feature_dict['n_nodes'] = len(list(tree.nodes()))
    
    # 计算内部节点数
    n_internal = 0
    for node in tree.nodes():
        if not tree.is_sample(node) and node != tree.root:
            n_internal += 1
    feature_dict['n_internal_nodes'] = n_internal
    
    feature_dict['n_sites'] = tree.num_sites
    
    # === 2. 分支长度特征 ===
    tbl = tree.total_branch_length
    feature_dict['total_branch_length'] = tbl
    
    branch_stats = compute_branch_length_stats(tree)
    feature_dict['mean_branch_length'] = branch_stats['mean']
    feature_dict['median_branch_length'] = branch_stats['median']
    feature_dict['std_branch_length'] = branch_stats['std']
    feature_dict['max_branch_length'] = branch_stats['max']
    
    # external_branch_prop
    external_bl = branch_stats['external_bl']
    internal_bl = branch_stats['internal_bl']
    feature_dict['internal_branch_length'] = internal_bl
    feature_dict['external_branch_prop'] = external_bl / tbl if tbl > 0 else 0.0
    
    # === 3. 合并时间特征 ===
    coal_stats = compute_coalescent_time_stats(tree, ts)
    feature_dict['tmrca'] = coal_stats['tmrca']
    feature_dict['mean_coal_time'] = coal_stats['mean']
    feature_dict['median_coal_time'] = coal_stats['median']
    feature_dict['std_coal_time'] = coal_stats['std']
    
    # === 4. 突变特征 ===
    n_mutations = tree.num_mutations
    feature_dict['n_mutations'] = n_mutations
    feature_dict['mutations_per_bp'] = n_mutations / tree_span if tree_span > 0 else 0.0
    feature_dict['mutations_per_tbl'] = n_mutations / tbl if tbl > 0 else 0.0
    
    # === 5. 等位基因频谱 ===
    afs_features = compute_afs_features(tree, n_samples)
    feature_dict['afs_singleton_prop'] = afs_features['singleton_prop']
    feature_dict['afs_doubleton_prop'] = afs_features['doubleton_prop']
    feature_dict['afs_rare_prop'] = afs_features['rare_prop']
    feature_dict['afs_low_freq_prop'] = afs_features['low_freq_prop']
    feature_dict['afs_common_prop'] = afs_features['common_prop']
    
    # === 6. 衍生特征 ===
    feature_dict['branch_length_cv'] = branch_stats['cv']
    feature_dict['time_skewness'] = coal_stats['skewness']
    feature_dict['diversity_theta'] = compute_diversity_theta(tree, n_samples)
    
    # === 元数据 ===
    feature_dict['tree_start'] = tree_start
    feature_dict['tree_end'] = tree_end
    
    # 按顺序组装特征向量
    feature_vector = []
    
    # 训练特征
    for feat_name in ALL_TRAINING_FEATURES:
        feature_vector.append(feature_dict[feat_name])
    
    # 元数据
    for feat_name in ALL_METADATA_FEATURES:
        feature_vector.append(feature_dict[feat_name])
    
    return np.array(feature_vector, dtype=np.float32)


# ============================================================================
# HDF5文件创建
# ============================================================================

def extract_tree_features_to_h5(
    ts_path: str,
    output_h5_path: str,
    chromosomes: Optional[List[str]] = None,
    overwrite: bool = False
):
    """
    从TreeSequence文件提取特征，保存到HDF5
    
    参数:
        ts_path: .trees文件路径或包含.trees文件的目录
        output_h5_path: 输出HDF5文件路径
        chromosomes: 要处理的染色体列表，如['chr1', 'chr2']
                    如果为None，自动检测ts_path中的所有.trees文件
        overwrite: 是否覆盖已存在的输出文件
    """
    
    # 检查输出文件
    if os.path.exists(output_h5_path) and not overwrite:
        raise FileExistsError(
            f"Output file {output_h5_path} already exists. "
            "Use --overwrite to replace it."
        )
    
    # 确定要处理的染色体
    ts_files = {}
    if os.path.isdir(ts_path):
        # 目录：自动检测.trees文件
        for file in os.listdir(ts_path):
            if file.endswith('.trees'):
                chrom = file.replace('.trees', '')
                if chromosomes is None or chrom in chromosomes:
                    ts_files[chrom] = os.path.join(ts_path, file)
    else:
        # 单个文件
        chrom = os.path.basename(ts_path).replace('.trees', '')
        ts_files[chrom] = ts_path
    
    if len(ts_files) == 0:
        raise ValueError(f"No .trees files found in {ts_path}")
    
    logger.info(f"Found {len(ts_files)} chromosomes to process: {list(ts_files.keys())}")
    
    # 创建HDF5文件
    with h5py.File(output_h5_path, 'w') as h5f:
        # === 全局元数据 ===
        h5f.attrs['creation_date'] = str(datetime.now())
        h5f.attrs['source_path'] = ts_path
        h5f.attrs['n_chromosomes'] = len(ts_files)
        h5f.attrs['feature_version'] = '1.0'
        
        # 特征名称
        h5f.attrs['training_feature_names'] = ALL_TRAINING_FEATURES
        h5f.attrs['metadata_feature_names'] = ALL_METADATA_FEATURES
        h5f.attrs['n_training_features'] = len(ALL_TRAINING_FEATURES)
        h5f.attrs['n_metadata_features'] = len(ALL_METADATA_FEATURES)
        
        # 特征分组信息
        for group_name, features in TREE_FEATURE_GROUPS.items():
            h5f.attrs[f'group_{group_name}'] = features
        
        # === 处理每条染色体 ===
        total_trees = 0
        total_bases = 0
        
        for chrom, ts_file in ts_files.items():
            logger.info(f"\nProcessing {chrom}...")
            logger.info(f"  Loading TreeSequence from {ts_file}")
            
            try:
                ts = tskit.load(ts_file)
            except Exception as e:
                logger.error(f"  Failed to load {ts_file}: {e}")
                continue
            
            genome_length = ts.sequence_length
            n_samples = ts.num_samples
            n_trees = ts.num_trees
            
            logger.info(f"  Genome length: {genome_length:,.0f} bp")
            logger.info(f"  Number of samples: {n_samples}")
            logger.info(f"  Number of trees: {n_trees:,}")
            
            # 提取所有树的特征
            intervals = []
            features = []
            
            for tree in tqdm(ts.trees(), total=n_trees, desc=f"  Extracting {chrom}"):
                interval = tree.interval
                intervals.append([interval.left, interval.right])
                
                feature_vector = extract_features_from_tree(tree, ts, genome_length)
                features.append(feature_vector)
            
            intervals = np.array(intervals, dtype=np.int64)
            features = np.array(features, dtype=np.float32)
            
            # 分离训练特征和元数据
            training_features = features[:, :TOTAL_FEATURES]
            metadata_features = features[:, TOTAL_FEATURES:]
            
            # 存储到HDF5
            grp = h5f.create_group(chrom)
            
            grp.create_dataset(
                'tree_intervals', 
                data=intervals,
                compression='gzip', 
                compression_opts=9
            )
            
            grp.create_dataset(
                'tree_features', 
                data=training_features,
                compression='gzip', 
                compression_opts=9
            )
            
            grp.create_dataset(
                'tree_metadata',
                data=metadata_features,
                compression='gzip',
                compression_opts=9
            )
            
            # 染色体级别的元数据
            grp.attrs['n_trees'] = n_trees
            grp.attrs['sequence_length'] = genome_length
            grp.attrs['n_samples'] = n_samples
            grp.attrs['first_tree_start'] = int(intervals[0, 0])
            grp.attrs['last_tree_end'] = int(intervals[-1, 1])
            
            total_trees += n_trees
            total_bases += genome_length
            
            logger.info(f"  ✓ {chrom}: {n_trees:,} trees, {TOTAL_FEATURES} features")
    
    # 最终统计
    file_size_mb = os.path.getsize(output_h5_path) / (1024 ** 2)
    
    logger.info("\n" + "="*70)
    logger.info("Feature extraction completed!")
    logger.info(f"  Output file: {output_h5_path}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Total chromosomes: {len(ts_files)}")
    logger.info(f"  Total trees: {total_trees:,}")
    logger.info(f"  Total bases: {total_bases:,}")
    logger.info(f"  Features per tree: {TOTAL_FEATURES} (training) + {len(ALL_METADATA_FEATURES)} (metadata)")
    logger.info("="*70)

# ============================================================================
# 阶段1: TreeSequence → NPZ (单染色体)
# ============================================================================

def extract_tree_features_to_npz(
    ts_path: str,
    output_npz_path: str,
    chrom_name: Optional[str] = None,
    sample_file: Optional[str] = None
) -> str:
    """
    从单个TreeSequence文件提取特征，保存为NPZ
    
    参数:
        ts_path: .trees文件路径
        output_dir: 输出目录
        chrom_name: 染色体名称（如'chr1'）。如果为None，从文件名推断
    
    返回:
        output_npz_path: 输出的NPZ文件路径
    """
    
    logger.info(f"Processing {chrom_name}...")
    logger.info(f"  Loading TreeSequence from {ts_path}")
    
    # 加载TreeSequence
    ts = tszip.load(ts_path) if ts_path.endswith('.tsz') else tskit.load(ts_path)
    # ========== 新增：样本筛选 ==========
    if sample_file is not None:
        logger.info(f"  Sample filtering enabled: {sample_file}")
        
        # 加载样本列表
        sample_ids = load_sample_list(sample_file)
        
        # 简化树序列
        ts, n_found, n_not_found = simplify_tree_sequence(ts, sample_ids)
        
        # 更新染色体名称（标记为筛选后）
        if chrom_name:
            chrom_name = f"{chrom_name}_filtered" 
    # =================================

    genome_length = ts.sequence_length
    n_samples = ts.num_samples
    n_trees = ts.num_trees
    
    logger.info(f"  Genome length: {genome_length:,.0f} bp")
    logger.info(f"  Number of samples: {n_samples}")
    logger.info(f"  Number of trees: {n_trees:,}")
    
    # 提取特征
    intervals = []
    features = []
    
    for tree in tqdm(ts.trees(), total=n_trees, desc=f"  Extracting {chrom_name}"):
        interval = tree.interval
        intervals.append([interval.left, interval.right])
        
        feature_vector = extract_features_from_tree(tree, ts, genome_length)
        features.append(feature_vector)
    
    intervals = np.array(intervals, dtype=np.int64)
    features = np.array(features, dtype=np.float32)
    
    # 分离训练特征和元数据
    TOTAL_FEATURES = len(get_all_feature_names(include_metadata=False))
    training_features = features[:, :TOTAL_FEATURES]
    metadata_features = features[:, TOTAL_FEATURES:]
    
    # 保存为NPZ
    
    np.savez_compressed(
        output_npz_path,
        tree_intervals=intervals,
        tree_features=training_features,
        tree_metadata=metadata_features,
        # 元信息
        chrom_name=chrom_name,
        n_trees=n_trees,
        sequence_length=genome_length,
        n_samples=n_samples,
        feature_names=get_all_feature_names(include_metadata=False),
        metadata_names=TREE_FEATURE_GROUPS['metadata']
    )
    
    file_size_mb = os.path.getsize(output_npz_path) / (1024 ** 2)
    logger.info(f"  ✓ Saved to {output_npz_path} ({file_size_mb:.2f} MB)")
    logger.info(f"  ✓ {n_trees:,} trees, {training_features.shape[1]} features\n")
    
    return output_npz_path


# ============================================================================
# 阶段2: NPZ → HDF5 (合并多染色体)
# ============================================================================

def convert_npz_to_h5(
    npz_files: List[str],
    output_h5_path: str,
    overwrite: bool = False
):
    """
    将多个NPZ文件合并为HDF5格式
    
    参数:
        npz_files: NPZ文件路径列表
        output_h5_path: 输出HDF5文件路径
        overwrite: 是否覆盖已存在的文件
    """
    # 检查输出文件
    if os.path.exists(output_h5_path) and not overwrite:
        raise FileExistsError(
            f"Output file {output_h5_path} already exists. "
            "Use overwrite=True to replace it."
        )
    
    logger.info(f"Converting {len(npz_files)} NPZ files to HDF5...")
    
    # 创建HDF5文件
    with h5py.File(output_h5_path, 'w') as h5f:
        # === 全局元数据 ===
        h5f.attrs['creation_date'] = str(datetime.now())
        h5f.attrs['n_chromosomes'] = len(npz_files)
        h5f.attrs['feature_version'] = '1.0'
        
        # 从第一个文件读取特征名称
        first_npz = np.load(npz_files[0], allow_pickle=True)
        feature_names = list(first_npz['feature_names'])
        metadata_names = list(first_npz['metadata_names'])
        
        h5f.attrs['training_feature_names'] = feature_names
        h5f.attrs['metadata_feature_names'] = metadata_names
        h5f.attrs['n_training_features'] = len(feature_names)
        h5f.attrs['n_metadata_features'] = len(metadata_names)
        
        # 特征分组信息
        for group_name, features in TREE_FEATURE_GROUPS.items():
            h5f.attrs[f'group_{group_name}'] = features
        
        # === 处理每个NPZ文件 ===
        total_trees = 0
        total_bases = 0
        
        for npz_path in npz_files:
            logger.info(f"\nProcessing {os.path.basename(npz_path)}...")
            
            # 加载NPZ
            data = np.load(npz_path, allow_pickle=True)
            
            chrom_name = str(data['chrom_name'])
            n_trees = int(data['n_trees'])
            sequence_length = float(data['sequence_length'])
            
            intervals = data['tree_intervals']
            training_features = data['tree_features']
            metadata_features = data['tree_metadata']
            
            logger.info(f"  Chromosome: {chrom_name}")
            logger.info(f"  Trees: {n_trees:,}")
            logger.info(f"  Sequence length: {sequence_length:,.0f}")
            
            # 创建染色体组
            grp = h5f.create_group(chrom_name)
            
            grp.create_dataset(
                'tree_intervals',
                data=intervals,
                compression='gzip',
                compression_opts=9
            )
            
            grp.create_dataset(
                'tree_features',
                data=training_features,
                compression='gzip',
                compression_opts=9
            )
            
            grp.create_dataset(
                'tree_metadata',
                data=metadata_features,
                compression='gzip',
                compression_opts=9
            )
            
            # 染色体元数据
            grp.attrs['n_trees'] = n_trees
            grp.attrs['sequence_length'] = sequence_length
            grp.attrs['n_samples'] = int(data['n_samples'])
            grp.attrs['first_tree_start'] = int(intervals[0, 0])
            grp.attrs['last_tree_end'] = int(intervals[-1, 1])
            
            total_trees += n_trees
            total_bases += sequence_length
            
            logger.info(f"  ✓ Added to HDF5")
    
    # 最终统计
    file_size_mb = os.path.getsize(output_h5_path) / (1024 ** 2)
    
    logger.info("\n" + "="*70)
    logger.info("Conversion completed!")
    logger.info(f"  Output file: {output_h5_path}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Total chromosomes: {len(npz_files)}")
    logger.info(f"  Total trees: {total_trees:,}")
    logger.info(f"  Total bases: {total_bases:,}")
    logger.info("="*70)

# ============================================================================
# 样本筛选函数（添加到辅助函数部分）
# ============================================================================

def load_sample_list(sample_file: str) -> List[str]:
    """
    从文件加载样本ID列表
    
    参数:
        sample_file: 样本ID文件路径（每行一个ID）
    
    返回:
        样本ID列表
    """
    with open(sample_file, 'r') as f:
        sample_ids = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(sample_ids)} sample IDs from {sample_file}")
    return sample_ids


def simplify_tree_sequence(
    ts: tskit.TreeSequence, 
    sample_ids: List[str]
) -> Tuple[tskit.TreeSequence, int, int]:
    """
    根据样本ID列表简化树序列
    
    参数:
        ts: 原始TreeSequence
        sample_ids: 要保留的样本ID列表
    
    返回:
        (简化后的TreeSequence, 找到的样本数, 未找到的样本数)
    """
    import json
    
    logger.info("Simplifying tree sequence based on sample list...")
    
    # 构建个体ID到样本节点的映射
    sample_nodes = []
    found_individuals = set()
    
    for ind in ts.individuals():
        # 解析元数据
        metadata = ind.metadata
        try:
            if isinstance(metadata, bytes):
                metadata_dict = json.loads(metadata.decode('utf-8'))
            else:
                metadata_dict = metadata
            
            individual_id = metadata_dict.get('individual_id', None)
            
            # 检查是否在目标列表中
            if individual_id and individual_id in sample_ids:
                sample_nodes.extend(ind.nodes)
                found_individuals.add(individual_id)
        
        except Exception as e:
            logger.warning(f"Failed to parse metadata for individual {ind.id}: {e}")
            continue
    
    # 统计
    n_found = len(found_individuals)
    n_not_found = len(sample_ids) - n_found
    
    if n_not_found > 0:
        unfound = set(sample_ids) - found_individuals
        logger.warning(f"  {n_not_found} sample IDs not found in tree sequence")
        if n_not_found <= 10:  # 只打印前10个
            logger.warning(f"  Unfound IDs: {unfound}")
    
    logger.info(f"  Found {n_found} individuals ({len(sample_nodes)} sample nodes)")
    
    # 简化
    sample_nodes = np.array(sample_nodes, dtype=np.int32)
    ts_simplified = ts.simplify(samples=sample_nodes, keep_input_roots=True)
    
    logger.info(f"  Original: {ts.num_samples} samples, {ts.num_trees} trees")
    logger.info(f"  Simplified: {ts_simplified.num_samples} samples, {ts_simplified.num_trees} trees")
    
    return ts_simplified, n_found, n_not_found

# ============================================================================
# 修改此函数的签名和开头部分
# ============================================================================

# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract tree features from TreeSequence files to HDF5 format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all .trees files in a directory
  python extract_tree_features.py --ts_dir /path/to/trees --output features.h5
  
  # Process specific chromosomes
  python extract_tree_features.py --ts_dir /path/to/trees --output features.h5 \\
      --chromosomes chr1 chr2 chr3
  
  # Process a single file
  python extract_tree_features.py --ts_path chr1.trees --output chr1_features.h5
        """
    )
    
    # 输入参数
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        '--ts_path',
        type=str,
        help='Path to a single .trees file'
    )
    
    # 输出参数
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output HDF5 file path'
    )
    
    # 可选参数
    parser.add_argument(
        '--chrom',
        type=str,
        default=None,
        help='Chromosome to process (e.g., chr1). If not specified, process all found .trees files'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output file if it exists'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--sample_file',
        type=str,
        default=None,
        help='Sample ID file (one ID per line). If provided, only these samples will be retained.'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    
    # 执行特征提取
    try:
        # extract_tree_features_to_h5(
        #     ts_path=ts_path,
        #     output_h5_path=args.output,
        #     chromosomes=args.chromosomes,
        #     overwrite=args.overwrite
        # )
        extract_tree_features_to_npz(
            ts_path=args.ts_path,
            output_npz_path=args.output,
            chrom_name=args.chrom if args.chrom else None,
            sample_file = args.sample_file
        )

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
