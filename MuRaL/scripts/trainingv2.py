import os
import sys
import time
import random
import warnings
from typing import Dict, Any, Union

import json
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from ray import tune

from pybedtools import BedTool

from scipy.stats import pearsonr


from MuRaL.data.data_preprocess_pipeline import DatasetPreprocessor
from MuRaL.utils.printer_utils import get_printer
from MuRaL.models.nn_models import *
from MuRaL.models.nn_utils import *
from MuRaL.evaluation.evaluation import *
from MuRaL.data.preprocessing import *
from MuRaL.data.dataset import dict_to_tuple_collate
from MuRaL.models.custom_loss import *
from MuRaL.models.losses import LossFactory, LossCalcStrategyFactory, NegativeBinomialLoss, DirichletMDNClassificationLoss, GammaMDNClassificationLoss
from MuRaL.training.optimizer import get_weight_decay, get_optimizer, get_lr_scheduler
from MuRaL.training.train import Trainer, TorchBackendManager, weights_init
from MuRaL.evaluation.observer import Observer, TimeMinor, GradMinor, LossMinor, DirMDNRecoder, GammaMDNRecoder
from MuRaL.utils.config_utils import read_bnn_config, read_feature_config

sys.path.append('/public/home/songhui/project/Mural/Mural_repo/MuRaL_112/model_utils')
from model_config import ModelFactory

warnings.filterwarnings('ignore',category=FutureWarning)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(config, args, checkpoint_dir=None):
    """
    Training funtion.
    
    Args:
        config: configuration of hyperparameters
        args: input args from the command line
        checkpoint_dir: checkpoint dir
    """

    ## bayesian model
    if args.use_bayesian:
        # dependency bayesian modules
        from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss
        from bayesian_torch.ao.quantization.quantize import enable_prepare, convert
        from bayesian_torch.models.bnn_to_qbnn import bnn_to_qbnn
        # compatibility check and read config
        assert int(args.model_no) in [127, 129, 132] , "Only model_no 127 (MuRaL_Hybrid) is supported for Bayesian training currently."
        const_bnn_prior_parameters = read_bnn_config(args.bnn_config)
        moped_enable = True if args.load_model_path else False
        const_bnn_prior_parameters['moped_enable'] = moped_enable


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
        'trial_dir' : args.trial_dir,
    }

    feature_config = read_feature_config(args.feature_config)
    # two sequence features must used: local and distal
    use_segment_task = True if (len(feature_config['features']) > 2 or args.use_segment_task) else False
    print("use_segment_task:", use_segment_task)

    preprocess_config.update(feature_config)
    preprocessor_pipline = DatasetPreprocessor(preprocess_config, use_h5=args.with_h5, printer=print)
    # 2026.0516
    calc_loss_strategy_name = args.calc_loss_strategy_name 
    # calc_loss_strategy_name = "AvgStepMutAndKmerMutUseInLocal" if args.calc_loss_strategy_name is None else args.calc_loss_strategy_name
    # (2025.12.18 to do): 根据config中是否包含sequence外的feature决定segment task是True or False
    dataset = preprocessor_pipline.preprocess_dataset(args.train_data, args.ref_genome, use_segment_task=use_segment_task)

    if args.validation_data:
        dataset_valid = preprocessor_pipline.preprocess_dataset(args.validation_data, args.ref_genome, use_segment_task=use_segment_task)
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
    segmentDatasetLoader_train = DataLoader(dataset_train, 1, shuffle=True, num_workers=segment_workers, pin_memory=False, collate_fn=dict_to_tuple_collate)
    dataloader_train = generate_data_batches(segmentDatasetLoader_train, config['sampled_segments'], config['batch_size'], shuffle=True, use_segment_task=use_segment_task)
        
    #dataloader_valid = generate_data_batches_v2(segmentDatasetLoader_valid, config['sampled_segments'], config['batch_size'], shuffle=False)
    segmentDatasetLoader_valid = DataLoader(dataset_valid, 1, shuffle=False, num_workers=segment_workers, pin_memory=False, collate_fn=dict_to_tuple_collate)
    dataloader_valid = generate_data_batches(segmentDatasetLoader_valid, config['sampled_segments'], config['batch_size'], shuffle=False, use_segment_task=use_segment_task)

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

    # model config
    # 2025.1.4, 考虑优化为自动生成或在config文件中定义
    config['no_of_cont'] = sum([feature.get('no_of_cont', 0) for f_name, feature in feature_config['features'].items()])

    emb_dims = [(x, min(16, int(x**0.25))) for x in dataset.cat_dims] 
    config['emb_dims'] = emb_dims 
    config['lin_layer_sizes']= [config['local_hidden1_size'], config['local_hidden2_size']]
    config['lin_layer_dropouts']=[config['local_dropout'], config['local_dropout']]
    config['n_class']=n_class
    config['emb_padding_idx'] = 4**config['local_order']
    # other config
    config['model_no'] = args.model_no
    config['without_bw_distal'] = args.without_bw_distal
    config['seq_only'] = args.seq_only
    config['restart_lr'] = args.restart_lr
    config['min_lr'] = args.min_lr
    model_factory = ModelFactory(config, args)
    model = model_factory.create_model(args.model_no)

    # model choice
    if args.load_model_path:
        model_load(model, args.load_model_path, freeze=True, device=device)
    else:
        model.apply(weights_init)
    if not args.use_bayesian:
        model.to(device)
    total_params = count_parameters(model)
    print("model:" )
    print(model)

    # loss and optimizer
    loss_factory = LossFactory()

    criterion = loss_factory.create_loss(
        loss_name=args.loss_name,
        use_sample_weight=args.recurrent,
        n_class=n_class,
    )
    is_nb = isinstance(criterion, NegativeBinomialLoss)
    if is_nb:
        if '_nb' not in str(args.model_no):
            raise ValueError(
                f"NegativeBinomialLoss requires an NB model variant "
                f"(e.g. 127_nb, 151_nb, 3_nb), but got model_no={args.model_no}"
            )
        criterion.to(device)
    is_dir_mdn = isinstance(criterion, DirichletMDNClassificationLoss)
    if is_dir_mdn:
        if '_dir_mdn' not in str(args.model_no):
            raise ValueError(
                f"DirichletMDN loss requires a DirMDN model variant "
                f"(e.g. 151_dir_mdn), but got model_no={args.model_no}"
            )
    is_gamma_mdn = isinstance(criterion, GammaMDNClassificationLoss)
    if is_gamma_mdn:
        if '_gamma_mdn' not in str(args.model_no):
            raise ValueError(
                f"GammaMDN loss requires a GammaMDN model variant "
                f"(e.g. 151_gamma_mdn), but got model_no={args.model_no}"
            )
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

    if args.use_bayesian:
        from MuRaL.training.train import BayesianTrainer
        config.update(const_bnn_prior_parameters)
        config['kl_loss'] = get_kl_loss
        config['kl_weight'] = args.kl_weight
        # wrap DNN to BNN, only support liner and conv layers
        dnn_to_bnn(model, const_bnn_prior_parameters)
        model.to(device)
        print("bnn model:" )
        print(model)
        trainer = BayesianTrainer(model, optimizer, scheduler, loss_calculator, criterion, device, config,
                                  observer=Observer, printer=print, train_strategy=calc_loss_strategy_name)
    else:
        trainer = Trainer(model, optimizer, scheduler, loss_calculator, criterion, device, config,
                          observer=Observer, printer=print, train_strategy=calc_loss_strategy_name,
                          collect_mu_r=is_nb)

    dir_mdn_recoder = DirMDNRecoder() if is_dir_mdn else None

    pi_entropy_recoder = GammaMDNRecoder() if is_gamma_mdn else None

    if not args.use_ray:
        early_stopping = EarlyStopping(patience=args.grace_period, verbose=True)

    # Training loop
    for epoch in range(args.epochs):
        epoch_time = time.time()
        save_path = get_save_path(args.use_ray, args.trial_dir, epoch)

        trainer.train_step(dataloader_train)
        if args.use_bayesian:
            valid_y_prob, valid_y_std = trainer.valid_step(dataloader_valid)
            valid_y_prob = to_np(valid_y_prob)
            print("valid_y_std 0:10 :", valid_y_std[:10])
        
        else:
            if is_dir_mdn:
                trainer.register_observer(dir_mdn_recoder)
            if is_gamma_mdn:
                trainer.register_observer(pi_entropy_recoder)
            valid_pred_y = trainer.valid_step(dataloader_valid)
            valid_y_prob = to_np(F.softmax(valid_pred_y, dim=1))
        valid_y = data_local_valid['mut_type'].to_numpy().squeeze()

        # Extract mu/r for NB models
        valid_mu, valid_r = None, None
        if is_nb:
            valid_mu_t, valid_r_t = trainer.get_mu_r()
            if valid_mu_t is not None:
                valid_mu = to_np(valid_mu_t)
                valid_r = to_np(valid_r_t)

        # Extract evidence for DirMDN models
        valid_evidence = None
        if is_dir_mdn:
            valid_evidence_t = dir_mdn_recoder.output()
            if valid_evidence_t is not None:
                valid_evidence = to_np(valid_evidence_t)
            dir_mdn_recoder.reset()
            trainer.remove_observer(dir_mdn_recoder)

        # Extract pi_entropy for GammaMDN models
        valid_pi_entropy = None
        if is_gamma_mdn:
            valid_pi_entropy_t = pi_entropy_recoder.output()
            if valid_pi_entropy_t is not None:
                valid_pi_entropy = to_np(valid_pi_entropy_t)
            pi_entropy_recoder.reset()
            trainer.remove_observer(pi_entropy_recoder)

        # calibrate
        if args.recurrent:
            weights = data_local_valid['sample_weight'].values.astype(int)
            indices = np.repeat(np.arange(len(weights)), np.maximum(weights, 1).astype(int))
            valid_y_prob_expanded = valid_y_prob[indices]
            valid_y_expanded = valid_y[indices]
            calibrator, fdiri_nll = calibrate_prob(valid_y_prob_expanded, valid_y_expanded, device, calibr_name='FullDiri')
        else:
            calibrator, fdiri_nll = calibrate_prob(valid_y_prob, valid_y, device, calibr_name='FullDiri')
        prob_cal = calibrator.predict_proba(valid_y_prob)

        if n_class == 7:
            n_sub = (7-1) // 2
            report_ac_prob_correlation(valid_y_prob, n_class=n_class, n_sub=n_sub)

        # Evaluation- Kmer
        if is_nb:
            evaluator_before_calibra = NBEvaluator(data_local_valid, valid_y_prob, n_class, mu=valid_mu, r=valid_r, use_obs_count=args.recurrent, printer=print)
            evaluator_after_calibra = NBEvaluator(data_local_valid, prob_cal, n_class, mu=valid_mu, r=valid_r, calibra="FullDiri", use_obs_count=args.recurrent, printer=print)
        elif is_gamma_mdn:
            evaluator_before_calibra = GammaMDNEvaluator(data_local_valid, valid_y_prob, n_class, pi_entropy=valid_pi_entropy, use_obs_count=args.recurrent, printer=print)
            evaluator_after_calibra = GammaMDNEvaluator(data_local_valid, prob_cal, n_class, pi_entropy=valid_pi_entropy, calibra="FullDiri", use_obs_count=args.recurrent, printer=print)
        elif is_dir_mdn:
            evaluator_before_calibra = DirMDNEvaluator(data_local_valid, valid_y_prob, n_class, evidence=valid_evidence, use_obs_count=args.recurrent, printer=print)
            evaluator_after_calibra = DirMDNEvaluator(data_local_valid, prob_cal, n_class, evidence=valid_evidence, calibra="FullDiri", use_obs_count=args.recurrent, printer=print)
        else:
            evaluator_before_calibra = Evaluator(data_local_valid, valid_y_prob, n_class, use_obs_count=args.recurrent, printer=print)
            evaluator_after_calibra = Evaluator(data_local_valid, prob_cal, n_class, calibra="FullDiri", use_obs_count=args.recurrent, printer=print)

        evaluator_before_calibra.evaluate_kmer()
        evaluator_after_calibra.evaluate_kmer()

        if is_nb:
            evaluator_before_calibra.evaluate_kmer_var(kmer_list=[3])
            evaluator_after_calibra.evaluate_kmer_var()

        if is_gamma_mdn:
            evaluator_before_calibra.evaluate_entropy_calibration()
            print("After calibration:")
            evaluator_after_calibra.evaluate_entropy_calibration()

        if is_dir_mdn:
            evaluator_before_calibra.evaluate_evidence_calibration()
            print("After calibration:")
            evaluator_after_calibra.evaluate_evidence_calibration()

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

        dataloader_train = generate_data_batches(segmentDatasetLoader_train, config['sampled_segments'], config['batch_size'], shuffle=True, use_segment_task=use_segment_task)
        dataloader_valid = generate_data_batches(segmentDatasetLoader_valid, config['sampled_segments'], config['batch_size'], shuffle=False, use_segment_task=use_segment_task)


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
    def __init__(self, data_local, y_prob, n_class, calibra=None, use_obs_count=False, printer=print):
        self.n_class = n_class
        self.prob_names = ['prob'+str(i) for i in range(n_class)]
        self.data_local = data_local
        self.y_prob = y_prob
        self.printer = printer
        self.calibra = calibra
        self.use_obs_count = use_obs_count
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
            kmer_corr = freq_kmer_comp_multi(self.data_and_prob, k, self.n_class, self.use_obs_count)
            self.printer(f"{k}{self.kmer_out_identify}", kmer_corr)
    
    def evaluate_regional_corr(self, chr_pos, win_size_list=[100000, 500000], save_valid_preds=False, save_path=None):
        cols = self.prob_names + (['sample_weight'] if self.use_obs_count else [])

        assert (chr_pos['mut_type'].astype(int).values == self.data_and_prob['mut_type'].astype(int).values).all(), \
            'ERROR: mut_type mismatch between position info and prediction data. ' \
                'BED file or data pipeline may have inconsistent ordering.'
        # if 'mut_type' not in cols:
            # cols = ['mut_type'] + cols

        valid_pred_df = pd.concat((chr_pos, self.data_and_prob[cols]), axis=1)
        # valid_pred_df.columns = ['chrom', 'start', 'end', 'strand'] + cols
        valid_pred_df.sort_values(['chrom', 'start'], inplace=True)
        valid_pred_df.reset_index(drop=True, inplace=True)

        if self.calibra is None:
            self.printer('valid_pred_df: ', valid_pred_df.head())

        for win_size in win_size_list:
            corr_win = corr_calc_sub(valid_pred_df, win_size, self.prob_names, self.use_obs_count)
            self.printer(self.regional_out_identify, str(win_size)+'bp', corr_win)

        if save_valid_preds:
            save_cols = ['chrom', 'start', 'end', 'strand', 'mut_type'] + self.prob_names
            valid_pred_df[save_cols].to_csv(save_path + '.valid_preds.tsv.gz', sep='\t', float_format='%.4g', index=False)

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
            corr_3mer = freq_kmer_comp_multi(self.data_and_prob.iloc[region_size*i: region_size*(i+1), ], 3, self.n_class, self.use_obs_count)
            corr_5mer = freq_kmer_comp_multi(self.data_and_prob.iloc[region_size*i: region_size*(i+1), ], 5, self.n_class, self.use_obs_count)

            score += np.sum([(1-corr)**2 for corr in corr_3mer]) + np.sum([(1-corr)**2 for corr in corr_5mer])

            avg_prob = calc_avg_prob(self.data_and_prob.iloc[region_size*i: region_size*(i+1), ], self.n_class, self.use_obs_count)
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


class NBEvaluator(Evaluator):
    """Evaluator 子类，增加 μ、r、方差评估。"""

    def __init__(self, data_local, y_prob, n_class, mu=None, r=None, calibra=None, use_obs_count=False, printer=print):
        super().__init__(data_local, y_prob, n_class, calibra=calibra, use_obs_count=use_obs_count, printer=printer)

        self.mu_names = ['mu'+str(i) for i in range(n_class)]
        self.r_names = ['r'+str(i) for i in range(n_class)]
        self.var_names = ['var'+str(i) for i in range(n_class)]

        if mu is not None:
            mu_df = pd.DataFrame(mu, columns=self.mu_names)
            r_df = pd.DataFrame(r, columns=self.r_names)
            # Var = μ + μ²/r
            var = mu + mu**2 / np.maximum(r, 1e-8)
            self.printer("calc mu by NegBinomial: ", mu.mean(axis=0))
            self.printer("calc r by NegBinomial: ", r.mean(axis=0))
            self.printer("calc var by NegBinomial: ", var.mean(axis=0))
            var_df = pd.DataFrame(var, columns=self.var_names)
            self.data_and_prob = pd.concat([self.data_and_prob, mu_df, r_df, var_df], axis=1)

    def evaluate_kmer_var(self, kmer_list=[3]):
        """按 k-mer 聚合 μ、r、方差，输出每类均值。"""
        if not hasattr(self, 'mu_names'):
            self.printer("NBEvaluator: no mu/r data, skip evaluate_kmer_var.")
            return

        for k in kmer_list:
            d = k // 2
            mer_list = ['us'+str(i) for i in range(d, 0, -1)] + ['ds'+str(i) for i in range(1, d+1)]

            grouped = self.data_and_prob.groupby(mer_list)
            result = grouped[self.mu_names + self.r_names + self.var_names].mean()

            self.printer(f"\n--- {k}mer μ/r/var ---")
            self.printer(result)


class DirMDNEvaluator(Evaluator):
    """Evaluator subclass for Dirichlet MDN models.

    Adds evidence-based reliability analysis:
    - Bin validation samples by evidence percentile
    - Report accuracy, confidence, and mean evidence per bin
    """

    def __init__(self, data_local, y_prob, n_class, evidence=None, calibra=None, use_obs_count=False, printer=print):
        super().__init__(data_local, y_prob, n_class, calibra=calibra, use_obs_count=use_obs_count, printer=printer)
        self.evidence = evidence

    def evaluate_evidence_calibration(self, n_bins=10):
        """Evaluate evidence-based reliability by percentile binning."""
        if self.evidence is None:
            self.printer("DirMDNEvaluator: no evidence data, skip evaluate_evidence_calibration.")
            return

        y_true = self.data_local['mut_type'].values
        y_pred = self.y_prob.argmax(axis=1)
        y_conf = self.y_prob.max(axis=1)

        bins = np.percentile(self.evidence, np.linspace(0, 100, n_bins + 1))
        bins[-1] += 1e-8

        bin_indices = np.digitize(self.evidence, bins) - 1

        results = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            acc = (y_pred[mask] == y_true[mask]).mean()
            conf = y_conf[mask].mean()
            ev = self.evidence[mask].mean()
            results.append((i, acc, conf, ev))

        self.printer(f"\n--- Evidence Calibration ({n_bins} bins) ---")
        self.printer(f"{'Bin':>5} {'Acc':>8} {'Conf':>8} {'Evidence':>10}")
        for i, acc, conf, ev in results:
            self.printer(f"{i:>5} {acc:>8.4f} {conf:>8.4f} {ev:>10.4f}")


class GammaMDNEvaluator(Evaluator):
    """Evaluator subclass for Gamma MDN models.

    For each mutation type c in {1,2,3} (skipping dominant class 0), computes
    observed density vs predicted mean correlation across pi_entropy bins.
    """

    def __init__(self, data_local, y_prob, n_class, pi_entropy=None, calibra=None,
                 use_obs_count=False, printer=print):
        super().__init__(data_local, y_prob, n_class, calibra=calibra,
                         use_obs_count=use_obs_count, printer=printer)
        self.pi_entropy = pi_entropy

    def evaluate_entropy_calibration(self, n_bins=10):
        """Evaluate per-mutation-type obs-density vs pred-mean correlation across entropy bins."""
        if self.pi_entropy is None:
            self.printer("GammaMDNEvaluator: no pi_entropy data, skip evaluate_entropy_calibration.")
            return None

        pi_entropy_np = self.pi_entropy
        if isinstance(pi_entropy_np, torch.Tensor):
            pi_entropy_np = pi_entropy_np.numpy()

        bin_edges = np.percentile(pi_entropy_np,
            np.linspace(0, 100, n_bins + 1))

        true_label = self.data_and_prob['mut_type'].values
        n_probs = self.y_prob.shape[1]

        records = []
        for i in range(n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (pi_entropy_np >= lo) & (pi_entropy_np <= hi)
            else:
                mask = (pi_entropy_np >= lo) & (pi_entropy_np < hi)

            n_samples = mask.sum()
            if n_samples == 0:
                continue

            avg_entropy = pi_entropy_np[mask].mean()
            label = (
                f"ent<{hi:.3f}" if i == 0 else
                f"ent>={lo:.3f}" if i == n_bins - 1 else
                f"ent=[{lo:.3f},{hi:.3f})"
            )

            record = {
                'bin': i,
                'bin_label': label,
                'n': n_samples,
                'entropy': avg_entropy,
            }

            for c in range(1, n_probs):
                obs_density = (true_label[mask] == c).mean()
                pred_mean = self.y_prob[mask, c].mean()
                record[f'prob{c}_obs_density'] = obs_density
                record[f'prob{c}_pred_mean'] = pred_mean

            records.append(record)

        self._print_bin_results(records, n_probs)

        correlations = {}
        for c in range(1, n_probs):
            obs_vals = [r[f'prob{c}_obs_density'] for r in records]
            pred_vals = [r[f'prob{c}_pred_mean'] for r in records]
            if len(obs_vals) >= 3:
                corr = np.corrcoef(obs_vals, pred_vals)[0, 1]
                correlations[f'prob{c}_corr'] = corr
                self.printer(f"  prob{c} obs vs pred correlation: {corr:.4f}")
            else:
                correlations[f'prob{c}_corr'] = None

        return correlations

    def _print_bin_results(self, records, n_probs):
        self.printer(f"\n--- Entropy Calibration ({len(records)} bins) ---")
        for row in records:
            parts = [
                f"  bin{row['bin']+1} {row['bin_label']}: "
                f"n={row['n']:>6d}  entropy={row['entropy']:.4f}"
            ]
            for c in range(1, n_probs):
                parts.append(
                    f"  prob{c}_obs={row[f'prob{c}_obs_density']:.6f}  "
                    f"prob{c}_pred={row[f'prob{c}_pred_mean']:.6f}"
                )
            self.printer(''.join(parts))


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

def report_ac_prob_correlation(valid_y_prob, n_class=7, n_sub=3):
    """
    Report correlation between AC=1 and AC>1 probabilities.

    Assumes class layout:
        [bg | AC=1 (n_sub) | AC>1 (n_sub)]
    """
    assert n_class == 1 + 2 * n_sub, \
        f"Expect n_class={1 + 2 * n_sub}, got {n_class}"

    ac1_start = 1
    acgt_start = 1 + n_sub

    # per-subtype correlation
    for i in range(n_sub):
        ac1_prob = valid_y_prob[:, ac1_start + i]
        acgt_prob = valid_y_prob[:, acgt_start + i]

        corr, _ = pearsonr(ac1_prob, acgt_prob)
        print(f"AC=1 and AC>1 subtype {i+1} correlation: {corr:.4f}")

    # summed correlation
    ac1_sum = valid_y_prob[:, ac1_start:ac1_start + n_sub].sum(axis=1)
    acgt_sum = valid_y_prob[:, acgt_start:acgt_start + n_sub].sum(axis=1)

    corr, _ = pearsonr(ac1_sum, acgt_sum)
    print(f"AC=1 and AC>1 probs sum correlation: {corr:.4f}")