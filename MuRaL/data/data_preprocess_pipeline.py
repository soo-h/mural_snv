import time

import pandas as pd
from Bio import SeqIO
from MuRaL.data.preprocessing import prepare_local_datav2, prepare_dataset_h5
from MuRaL.data.segment_preprocessing import prepare_soft_label, prepare_soft_label2, prepare_soft_label3, prepare_soft_labelv2,prepare_soft_label2v2 
from MuRaL.data.dataset import CombinedDatasetNPv2
from MuRaL.data.prepare_refseq_information import compute_nuc_skew

from MuRaL.data.map_segment_feature import prepare_segment_feature

from pybedtools import BedTool
import os
import pickle

import warnings
from typing import Dict, Any, Callable, List, Union
from pathlib import Path
import numpy as np

def prepare_dataset_npv3(bed_regions, ref_genome, bw_files, bw_names, bw_radii,central_radius=30000, local_radius=5, local_order=1, distal_radius=50, distal_order=1, seq_only=False, 
                         without_bw_distal=False, segment_task=False, distal_encoding=None, segment_calc_method=None, path_type=None, prediction=False, segment_info_length=None,
                         single_base_task=False, segment_length_config=None, **kwargs):
    """Prepare the datasets for given regions, without an H5 file
    Input: bed_regions, ref_genome, config
    │
    ├─→ DataLoader
    │     ├─→ prepare_local_datav2()
    │     └─→ get_distal_seqs_by_region()
    │           │
    │           └─→ base_data: DataFrame
    │
    ├─→ FeatureFactory
    │     ├─→ parse config.bigwig_files
    │     │     └─→ BigWigFeatureSource x N
    │     │
    │     ├─→ parse config.features['segment_avg_mut']
    │     │     └─→ PrecomputedFeatureSource
    │     │           ├─→ load from disk
    │     │           └─→ validate shapes
    │     │
    │     └─→ parse config.features['gc_content']
    │           └─→ ComputedFeatureSource
    │                 └─→ register compute_fn
    │
    └─→ DatasetBuilder
          ├─→ receive base_data
          ├─→ receive feature_sources
          ├─→ compute n_channels (内部计算)
          └─→ build CombinedDatasetNPv2
                    │
                    └─→ Output: Dataset
    """
    # Prepare local data
    ref_genome = SeqIO.to_dict(SeqIO.parse(open(ref_genome, 'r'), 'fasta'))
    data_local, seq_cols, categorical_features, col_name_label = prepare_local_datav2(bed_regions, ref_genome, bw_files, bw_names, bw_radii, central_radius, local_radius, local_order, seq_only)

    features = FeatureFactory().create_all(kwargs['config'])

    # Combine local data and distal into Dataset objects  
    dataset = CombinedDatasetNPv2(
        data=data_local, 
        seq_cols=seq_cols, 
        cat_cols=categorical_features, 
        output_col=col_name_label, 
        ref_genome=ref_genome, 
        bed_regions=bed_regions, 
        central_radius=central_radius, 
        distal_radius=distal_radius, 
        distal_order=distal_order, 
        seq_only=seq_only, 
        distal_encoding=distal_encoding, 
        segment_calc_method=segment_calc_method, 
        feature_sources
        )

    return dataset, segment_task




class FeatureFactory:
    """
    只包含两种特征读取方式（可能并非都存在）：
    1. BigWig 文件 (bw_files)
    2. 实时生成特征（to do: for distal sequence feature）
    3. 预计算特征
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: {
                'features': {
                    'step_avg_strategy': {
                        'type': 'bigwig',
                        'path': '/path/to/file.bw'
                    },
                    'step_avg_kmer_mut': {
                        'type': 'bigwig',
                        'path': '/path/to/file.bw'
                    },
                    'nuc_skew': {
                        'type': 'computed',
                    }
                }
            }
        """
        self.config = config
        self._registry = {
            'bigwig':self._create_bigwig_feature,
            'computed': self._create_computed_feature,
            }
    
    
    def create_all(self):
        features = {}
        for name, feature_config in self.config.features.items():
            feature_type = feature_config['type']
            creator = self._registry[feature_type]
            features[name] = creator(name, feature_config)
        return features
    def _create_bigwig_feature(self, name: str, config: Dict[str, Any]) -> 'BigWigFeatureSource':
        """
        创建 BigWig 特征源
        
        config: {
            'type': 'bigwig',
            'path': '/path/to/file.bw'
        }
        """
        path = config.get('path')
        if not path:
            raise ValueError(f"BigWig feature '{name}' missing 'path'")
        
        return BigWigFeatureSource(path)
    
    def _create_computed_feature(self, name: str, config: Dict[str, Any]) -> 'ComputedFeatureSource':
        """
        创建实时计算特征源
        
        config: {
            'type': 'computed',
            'compute_fn': 'compute_nuc_skew'
        }
        """
        compute_fn_registry = {
            'nuc_skew': compute_nuc_skew,
        }

        compute_fn_name = config.get('compute_fn')
        params = config.get('params', {})
        compute_fn = compute_fn_registry.get(compute_fn_name)
        if not compute_fn:
            raise ValueError(f"Computed feature '{name}' missing 'compute_fn'")
        
        return ComputedFeatureSource(compute_fn, params)

        single_base_task = prepare_single_base_info(
            bed_regions, 
            central_radius, 
            ref_genome, 
            single_base_task_config)
    
class FeatureSource:
    @property
    def n_channels(self) -> int:
        raise NotImplementedError
    
    def get(self, region, index: int):
        raise NotImplementedError


class BigWigFeatureSource(FeatureSource):
    def __init__(self, bw_path: str):
        if not os.path.exists(bw_path):
            raise FileNotFoundError(f"BigWig file not found: {bw_path}")
        
        import pyBigWig
        self.bw = pyBigWig.open(bw_path)
        self.bw_path = bw_path
        self._n_channels = 1
    
    @property
    def n_channels(self) -> int:
        return self._n_channels
    
    def get(self, region, index: int):
        values = self.bw.values(region.chrom, region.start, region.end)
        return np.array(values, dtype=np.float32)
    
    def __repr__(self):
        return f"BigWigFeatureSource({Path(self.bw_path).name})"


class ComputedFeatureSource(FeatureSource):
    def __init__(self, compute_fn: Callable, params: Dict[str, Any]):
        if not callable(compute_fn):
            raise TypeError(f"compute_fn must be callable, got {type(compute_fn)}")
        
        self.compute_fn = compute_fn
        self.params = params
        self._features = self.conpute_fn(**params)
        self._n_channels = params.get('n_channels')
        self._first_computed = False
    
    @property
    def n_channels(self) -> int:
        if self._n_channels is None:
            return 1  # 占位，第一次 get() 时确定
        return self._n_channels
    
    def get(self, region, index: int):
        return self._features[index]
    
    def __repr__(self):
        fn_name = getattr(self.compute_fn, '__name__', str(self.compute_fn))
        return f"ComputedFeatureSource(fn={fn_name})"


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