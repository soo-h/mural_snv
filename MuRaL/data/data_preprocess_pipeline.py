import time
import copy

import pandas as pd
from Bio import SeqIO
from MuRaL.data.preprocessing import prepare_dataset_h5
from MuRaL.data.segment_preprocessing import prepare_soft_label, prepare_soft_label2, prepare_soft_label3, prepare_soft_labelv2,prepare_soft_label2v2 
from MuRaL.data.dataset import CombinedDatasetNPv2
from MuRaL.data.prepare_refseq_information import compute_nuc_skew
from MuRaL.data.preprocessing import distal_encoding_by_region, annot_encoding_by_region, get_distal_seqs_by_segments , kmer_encoding_by_region, bpe_encoding_by_region, bed_reader, prepare_local_feature

from MuRaL.data.map_segment_feature import prepare_segment_feature

from pybedtools import BedTool
import os
import pickle

import warnings
from typing import Dict, Any, Callable, List, Union
from pathlib import Path
import numpy as np


def prepare_dataset_npv3(segments, ref_genome,  **kwargs):

    # Prepare local data
    ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    features = FeatureFactory(segments, ref_genome, kwargs['config']).create_all()

    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNPv2(
        segments=segments, 
        features = features,
        features_without_train = kwargs['config'].get('features_without_train', ['local_seq']),
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
    return features[indices]

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

class PrecomputedFeatureSource(FeatureSource):
    def __init__(self, features: np.ndarray):
        self.features = features
        if isinstance(features[0], (int, float)):
            value = features[0]
        else:
            value = features[0][0]
        if isinstance(value, (list, np.ndarray)):
            self._n_channels = len(value)
        elif isinstance(value, (int, float)):
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
    def __init__(self, preprocess_config, use_h5, printer=print):
        self.config = preprocess_config
        self.use_h5 = use_h5
        self.printer = print

    def preprocess_dataset(self, bed_path, ref_genome, use_segment_task=False):
        bed = self.read_bed_file(bed_path)
        bed_generator = bed_reader(bed, self.config.get('segment_center'))
        bed_regions = [(batch, stand) for batch, stand in bed_generator]

        bw_files, bw_names, bw_radii = self.get_bw_paths()

        if self.use_h5:
            return self._process_h5(bed, ref_genome, bw_files, bw_names, bw_radii, use_segment_task)
        else:
            return self._process_np(bed_regions, ref_genome, self.config)

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


    def _process_np(self, bed_regions, ref_genome, config):
        # Non-H5 logic
        self.printer('using numpy/pandas for distal_seq ...')
        step_stime = time.time()
        dataset, segment_task = prepare_dataset_npv3(bed_regions, ref_genome, self.config)

        #if segment_task and not prediction:
            #self._save_segment_task(segment_task, self.config['trial_dir'])

        self.printer(f"preprocess without H5 used time:", (time.time() - step_stime))
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