import warnings

import torch.backends
warnings.filterwarnings('ignore',category=FutureWarning)


from pybedtools import BedTool

import sys
sys.path.append('/public/home/songhui/project/Mural/Mural_repo/MuRaL_112/model_utils')
from model_config import ModelFactory

from MuRaL.data.data_preprocess_pipeline import DatasetPreprocessor

import argparse
import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data import random_split


from functools import partial
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import os
import time
import datetime
import random

from MuRaL.utils.printer_utils import get_printer
from MuRaL.models.nn_models import *
from MuRaL.models.nn_utils import *
from MuRaL.evaluation.evaluation import *
from MuRaL.data.custom_dataloader import MyDataLoader
from MuRaL.data.preprocessing import *

from MuRaL.models.custom_loss import *
from MuRaL.models.losses import LossFactory, LossCalcStrategyFactory
from MuRaL.training.optimizer import get_weight_decay, get_optimizer, get_lr_scheduler
from MuRaL.training.train import Trainer, TorchBackendManager, weights_init

from MuRaL.evaluation.observer import Observer, TimeMinor, GradMinor, LossMinor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def adapt_calc_loss_strategy(strategy):
    if strategy == 'AvgSegMutUseInLocalv2':
        return 'AvgSegMutUseInLocal'
    return strategy
def get_calc_segment_info_method(calc_strategy, print=print):
    if calc_strategy == 'AvgSegmentLabel_withGAN' or calc_strategy == 'AvgSegmentLabel_withGAN2':
        method = 'SegMut'
    elif calc_strategy == 'AvgSegMutUseInLocal' or calc_strategy == 'AvgSegMutAndNucSkewUseInLocal':
        method = 'SegMutRate'
    elif calc_strategy == 'AvgSegAndKmerMut' or calc_strategy == 'AvgSegMutAndKmerMutUseInLocal':
        method = 'AvgSegMutAndKmerMut'
    elif calc_strategy == 'AvgSegMutUseInLocalv2':
        method = 'SegMutRateByRegion'
    elif calc_strategy == 'AvgStepMutAndKmerMutUseInLocal' or calc_strategy == 'AvgStepMutAndKmerMutCominedLoss':
        method = 'AvgStepMutAndKmerMut'
    elif calc_strategy == 'AvgStepMutUseInLocal':
        method = 'AvgStepMut'
    else:
        method = None

    print("Segment Info utils strategy: ", method)
    return method

def train(config, args, checkpoint_dir=None):
    """
    Training funtion.
    
    Args:
        config: configuration of hyperparameters
        args: input args from the command line
        checkpoint_dir: checkpoint dir
    """

    start_time = time.time()
    args.segment_task = None
    n_class = args.n_class

    if not args.use_ray:
        print = get_printer(args.use_ray, args.trial_training_log)
    
    torch_backend_manager = TorchBackendManager(args.use_dilation, args.cudnn_benchmark_false, printer=print)
    torch_backend_manager.set_torch_backends()
    torch_backend_manager.display_torch_device_info()

    # prepare dataset
    # Note, same key in args and config has different data type
    preprocess_config = {
        'segment_center': config['segment_center'],
        'local_radius' : config['local_radius'],
        'local_order' : config['local_order'],
        'distal_radius' : config['distal_radius'],
        'distal_order' : args.distal_order,
        'h5f_path' : args.h5f_path,
        'seq_only' : args.seq_only,
        'n_h5_files' : args.n_h5_files,
        'without_bw_distal' : args.without_bw_distal,
        'bw_paths' : args.bw_paths,
        'segment_length_config' : args.use_segment_length_config,
        'trial_dir' : args.trial_dir,
        'slid_strategy' : args.sliding_strategy,
        'step_avg_strategy': args.step_avg_strategy
    }

    preprocessor_pipline = DatasetPreprocessor(preprocess_config, use_h5=args.with_h5, printer=print)
    segment_calc_method = get_calc_segment_info_method(args.calc_loss_strategy_name, print=print)
    calc_loss_strategy_name = adapt_calc_loss_strategy(args.calc_loss_strategy_name)
    print("single_base_task:", args.use_single_base_task)
    dataset = preprocessor_pipline.preprocess_dataset(args.train_data, args.ref_genome, use_segment_task=args.use_segment_task, distal_encoding=args.distal_encoding, segment_calc_method=segment_calc_method, path_type=args.path_type, single_base_task=args.use_single_base_task)

    if args.validation_data:
        dataset_valid = preprocessor_pipline.preprocess_dataset(args.validation_data, args.ref_genome, use_segment_task=args.use_segment_task, distal_encoding=args.distal_encoding, segment_calc_method=segment_calc_method, path_type=args.path_type, single_base_task=args.use_single_base_task)
        dataset_train = dataset
    else:
        print("Error: validation should provided.")
    
    data_local = dataset_train.data_local.reset_index(drop=True)
    data_local_valid = dataset_valid.data_local.reset_index(drop=True)
    train_size = len(data_local)
    valid_size = len(data_local_valid)
    print("train_size, valid_size: ", train_size, valid_size)

    # data loader
    segment_workers = args.cpu_per_trial - 1
    #dataloader_train = generate_data_batches_v2(segmentDatasetLoader_train, config['sampled_segments'], config['batch_size'], shuffle=True)
    segmentDatasetLoader_train = DataLoader(dataset_train, 1, shuffle=True, num_workers=segment_workers, pin_memory=False)
    dataloader_train = generate_data_batches(segmentDatasetLoader_train, config['sampled_segments'], config['batch_size'], shuffle=True, use_segment_task=args.use_segment_task)
        
    #dataloader_valid = generate_data_batches_v2(segmentDatasetLoader_valid, config['sampled_segments'], config['batch_size'], shuffle=False)
    segmentDatasetLoader_valid = DataLoader(dataset_valid, 1, shuffle=False, num_workers=segment_workers, pin_memory=False)
    dataloader_valid = generate_data_batches(segmentDatasetLoader_valid, config['sampled_segments'], config['batch_size'], shuffle=False, use_segment_task=args.use_segment_task)

    # get device
    device = torch.device('cpu')
    if args.gpu_per_trial > 0:
        # Set the device
        if not torch.cuda.is_available():
            print('Warning: You requested GPU computing, but CUDA is not available! If you want to run without GPU, please set "--ray_ngpus 0 --gpu_per_trial 0"', file=sys.stderr)
        if not args.use_ray:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
                torch.cuda.set_device(f'cuda:{args.cuda_id}')
        else:    
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model choice
    # model config
    config['no_of_cont'] = len(dataset.cont_cols)
    if calc_loss_strategy_name == 'AvgSegMutUseInLocal' or calc_loss_strategy_name == 'AvgSegMutAndNucSkewUseInLocal' or calc_loss_strategy_name == 'AvgSegMutAndKmerMutUseInLocal' or calc_loss_strategy_name == 'AvgStepMutAndKmerMutUseInLocal' \
        or calc_loss_strategy_name == 'AvgStepMutUseInLocal':
        config['no_of_cont'] += 3

    emb_dims = [(x, min(16, int(x**0.25))) for x in dataset.cat_dims] 
    config['emb_dims'] = emb_dims 
    config['lin_layer_sizes']= [config['local_hidden1_size'], config['local_hidden2_size']]
    config['lin_layer_dropouts']=[config['local_dropout'], config['local_dropout']]
    config['avg_mut_dropout'] = config['avg_mut_dropout']
    config['n_class']=n_class
    config['emb_padding_idx'] = 4**config['local_order']
    config['n_class'] = n_class
    # other config
    config['model_no'] = args.model_no
    config['without_bw_distal'] = args.without_bw_distal
    config['seq_only'] = args.seq_only
    config['restart_lr'] = args.restart_lr
    config['min_lr'] = args.min_lr
    model_factory = ModelFactory(config, args)
    model = model_factory.create_model(args.model_no)

    if args.load_model_path:
        model_load(model, args.load_model_path, freeze=True, device=device)
    else:
        model.apply(weights_init)

    model.to(device)
    total_params = count_parameters(model)
    print("model:" )
    print(model)

    # loss and optimizer
    loss_factory = LossFactory()
    criterion = loss_factory.create_loss()
    loss_calculator = LossCalcStrategyFactory.get_loss_strategy(calc_loss_strategy_name, avg_mut_loss_strategy=args.mix_loss)

    config['weight_decay'] = get_weight_decay(config['batch_size'], args.epochs, train_size, args.weight_decay_auto, config['weight_decay']) 
    optimizer = get_optimizer(config['optim'], model, config['learning_rate'], config['weight_decay'])
    scheduler = get_lr_scheduler(config['lr_scheduler'], optimizer, train_size, config)

    print('optimizer:', optimizer)
    print('scheduler:', scheduler)
    sys.stdout.flush()

    chr_pos = get_position_info(BedTool(args.validation_data), config['segment_center'])

    Observer = [TimeMinor(out_after_n_batch=1000, printer=print), 
                GradMinor(out_after_n_batch=1000, printer=print), 
                LossMinor(calc_loss_strategy_name, printer=print)]
    trainer = Trainer(model, optimizer, scheduler, loss_calculator, criterion, device, config, 
                      observer=Observer, printer=print, train_strategy=calc_loss_strategy_name)

    if not args.use_ray:
        early_stopping = EarlyStopping(patience=args.grace_period, verbose=True)

    # Training loop
    for epoch in range(args.epochs):
        epoch_time = time.time()
        save_path = get_save_path(args.use_ray, args.trial_dir, epoch)

        trainer.train_step(dataloader_train)
        valid_pred_y = trainer.valid_step(dataloader_valid)
        valid_y_prob = to_np(F.softmax(valid_pred_y, dim=1))
        valid_y = data_local_valid['mut_type'].to_numpy().squeeze()

        # calibrate
        calibrator, fdiri_nll = calibrate_prob(valid_y_prob, valid_y, device, calibr_name='FullDiri')
        prob_cal = calibrator.predict_proba(valid_y_prob)

        # Evaluation- Kmer
        evaluator_before_calibra = Evaluator(data_local_valid, valid_y_prob, n_class, printer=print)
        evaluator_after_calibra = Evaluator(data_local_valid, prob_cal, n_class, calibra="FullDiri", printer=print)

        evaluator_before_calibra.evaluate_kmer()
        evaluator_after_calibra.evaluate_kmer()

        evaluator_before_calibra.evaluate_regional_score(valid_size)
        evaluator_after_calibra.evaluate_regional_score(valid_size)

        evaluator_before_calibra.evaluate_regional_corr(chr_pos, save_valid_preds=args.save_valid_preds, save_path=save_path)
        evaluator_after_calibra.evaluate_regional_corr(chr_pos)

        minor_metrics = {
            'score_before_calibra': evaluator_before_calibra.metrics['score'],
            'current_valid_loss' : trainer.metrics['valid_loss'],
            'current_valid_fdiri_loss' : fdiri_nll,
        }

        if epoch == 0:
            min_loss = minor_metrics['current_valid_loss']
            after_min_loss = 0
            min_loss_epoch = 0
        else:
            if minor_metrics['current_valid_loss'] < min_loss:
                min_loss = minor_metrics['current_valid_loss']
                min_loss_epoch = epoch
            else:
                after_min_loss = epoch - min_loss_epoch
        
        non_ray_checkpoint_dir = f'{args.trial_dir}/checkpoint_{epoch}'
        save_model_metrics(args.use_ray, 
                           non_ray_checkpoint_dir, 
                           epoch, 
                           after_min_loss, 
                           total_params,
                           minor_metrics)

        save_model(model, 
                   calibrator, 
                   config, 
                   save_path)

        if not args.use_ray:
            early_stopping(minor_metrics['current_valid_loss'], model)
            if early_stopping.early_stop:
                print(f"Epoch {epoch} used time:{time.time()-epoch_time} seconds!")
                print("Early stopping")
                break

        if config['lr_scheduler'] == 'ROP':
            scheduler.step(minor_metrics['current_valid_loss'])
        torch.cuda.empty_cache() 
            
        print(f"Epoch {epoch} used time:{time.time()-epoch_time} seconds!")
        sys.stdout.flush()

        dataloader_train = generate_data_batches(segmentDatasetLoader_train, config['sampled_segments'], config['batch_size'], shuffle=True, use_segment_task=args.use_segment_task)
        dataloader_valid = generate_data_batches(segmentDatasetLoader_valid, config['sampled_segments'], config['batch_size'], shuffle=False, use_segment_task=args.use_segment_task)


    print(f"dital_radius: {args.distal_radius} training finish, {epoch} epochs total time:{time.time()-start_time} min!")
    best_epoch = epoch - early_stopping.counter
    print(f"Best Epoch: {best_epoch}")

def freeze_without_large_scale_model(model):
    for name, param in model.named_parameters():
        if 'large_scale_model' not in name:
            param.requires_grad = False
            print(f'Freeze parameter: {name}')

def model_load(model, load_path, freeze, device):
    model_state = torch.load(load_path, map_location=device)
    model.load_state_dict(model_state)
    if freeze:
        freeze_without_large_scale_model(model)

def save_model_metrics(use_ray, non_ray_checkpoint_dir, epoch, after_min_loss, total_params, minor_metrics):
    if use_ray:
        tune.report(loss=minor_metrics['current_valid_loss'], 
                    fdiri_loss=minor_metrics['current_valid_fdiri_loss'], 
                    after_min_loss=after_min_loss, 
                    score=minor_metrics['score_before_calibra'], 
                    total_params=total_params)
    else:
        metrics = {
            'loss': minor_metrics['current_valid_loss'],
            'fdiri_loss': minor_metrics['current_valid_fdiri_loss'],
            'after_min_loss': after_min_loss,
            'score': minor_metrics['score_before_calibra'],
            'total_params': total_params}
        
        report_path = os.path.join(non_ray_checkpoint_dir, f'epoch_{epoch}_metrics.txt')
        report_metrics(metrics, report_path)

        
def get_save_path(use_ray, trial_dir, epoch):
    if use_ray:
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'model')
    else:
        # Define a directory to save the files when not using Ray
        non_ray_checkpoint_dir = f'{trial_dir}/checkpoint_{epoch}'
        os.makedirs(non_ray_checkpoint_dir, exist_ok=True)
        path = os.path.join(non_ray_checkpoint_dir, 'model')
    return path

class Evaluator:
    def __init__(self, data_local, y_prob, n_class, calibra=None, printer=print):
        self.n_class = n_class
        self.prob_names = ['prob'+str(i) for i in range(n_class)]
        self.data_local = data_local
        self.y_prob = y_prob
        self.printer = printer
        self.calibra = calibra
        self.data_and_prob = self.preprocess()
        self.kmer_out_identify, self.regional_out_identify = self.set_output_identifiers()
        self.metrics = {}
    
    def preprocess(self):
        y_prob = pd.DataFrame(data=np.copy(self.y_prob), columns=self.prob_names)
        data_and_prob = pd.concat([self.data_local, y_prob], axis=1)
        return data_and_prob

    def set_output_identifiers(self):

        kmer_out_identify = {
            'no_calibra' : 'mer correlation - all: ',
            'FullDiri' : 'mer correlation(after fdiri_cal)'
        }

        regional_out_identify = {
            'no_calibra' : 'regional corr (validation):',
            'FullDiri' : 'regional corr (validation, after calibration):'
        }

        if self.calibra is None:
            return kmer_out_identify['no_calibra'], regional_out_identify['no_calibra']
        return kmer_out_identify['FullDiri'], regional_out_identify['FullDiri']

    def evaluate_kmer(self, kmer_list=[3,5,7]):
        if self.calibra is None:
            self.printer("valid_data_and_prob.iloc[0:10]", self.data_and_prob.iloc[0:10])
        for k in kmer_list:
            kmer_corr = freq_kmer_comp_multi(self.data_and_prob, k, self.n_class)
            self.printer(f"{k}{self.kmer_out_identify}", kmer_corr)
    
    def evaluate_regional_corr(self, chr_pos, win_size_list=[100000, 500000], save_valid_preds=False, save_path=None):
        valid_pred_df = pd.concat((chr_pos, self.data_and_prob[['mut_type'] + self.prob_names]), axis=1)
        valid_pred_df.columns = ['chrom', 'start', 'end', 'strand', 'mut_type'] + self.prob_names
        valid_pred_df.sort_values(['chrom', 'start'], inplace=True)
        valid_pred_df.reset_index(drop=True, inplace=True)

        if self.calibra is None:
            self.printer('valid_pred_df: ', valid_pred_df.head())

        for win_size in win_size_list:
            corr_win = corr_calc_sub(valid_pred_df, win_size, self.prob_names)
            self.printer(self.regional_out_identify, str(win_size)+'bp', corr_win)
        
        if save_valid_preds:
            valid_pred_df.to_csv(save_path + '.valid_preds.tsv.gz', sep='\t', float_format='%.4g', index=False)

    def evaluate_regional_score(self, valid_size):
        if valid_size > 10000 * 10:
            region_size = 10000
        else:
            region_size = valid_size // 10
        n_regions = valid_size // region_size
        self.printer('n_regions:', n_regions)

        score = 0
        corr_3mer = []
        corr_5mer = []
            
        region_avg = []
        for i in range(n_regions):
            corr_3mer = freq_kmer_comp_multi(self.data_and_prob.iloc[region_size*i: region_size*(i+1), ], 3, self.n_class)    
            corr_5mer = freq_kmer_comp_multi(self.data_and_prob.iloc[region_size*i: region_size*(i+1), ], 5, self.n_class)
                
            score += np.sum([(1-corr)**2 for corr in corr_3mer]) + np.sum([(1-corr)**2 for corr in corr_5mer])
                
            avg_prob = calc_avg_prob(self.data_and_prob.iloc[region_size*i: region_size*(i+1), ], self.n_class)
            region_avg.append(avg_prob)
            #print("avg_prob:", avg_prob, i)
            
        region_avg = pd.DataFrame(region_avg)
        #print('region_avg.head():', region_avg.head())
        corr_list = []
        for i in range(self.n_class):
            corr_list.append(region_avg[i].corr(region_avg[i + self.n_class]))

        if self.calibra is None:    
            self.printer('corr_list:', corr_list)
            #print('corr_3mer:', corr_3mer)
            #print('corr_5mer:', corr_5mer)
            self.printer('regional score:', score, n_regions)
        else:
            self.printer('corr_list(after fdiri_cal)', corr_list)
            self.printer('regional score(after fdiri_cal)', score, n_regions)
        
        self.metrics['score'] = score
        

def save_model(model, fdiri_cal, config, save_path):
    """Save model state, fdiri_cal, config and validation predictions to the specified path."""
    torch.save(model.state_dict(), save_path)

    with open(save_path + '.fdiri_cal.pkl', 'wb') as pkl_file:
        pickle.dump(fdiri_cal, pkl_file)

    with open(save_path + '.config.pkl', 'wb') as fp:
        pickle.dump(config, fp)

def report_metrics(metrics, report_path=None):
    """Report metrics by saving them to a file or printing them."""
    if report_path:
        with open(report_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
    else:
        for key, value in metrics.items():
            print(f"{key}: {value}")