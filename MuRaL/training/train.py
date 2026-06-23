import logging
import torch
import time
import sys

from dataclasses import dataclass

logger = logging.getLogger('mural')
from typing import Dict, Tuple, Any, Callable
from enum import Enum, auto

import torch.nn as nn
import torch.nn.functional as F
from MuRaL.evaluation.observer import Observer, TimeMinor, GradMinor, LossMinor, PredsRecoder, MuRRecoder, ContributionMinor2

class TrainerSubject:
    def __init__(self):
        self.observers = []
        self.metrics = {}

    def register_observer(self, observer: Observer):
        self.observers.append(observer)

    def remove_observer(self, observer: Observer):
        self.observers.remove(observer)

    def notify_observers(self, **kwargs):
        for observer in self.observers:
            indicator = observer.update(**kwargs)
            if indicator:
                self.metrics.update(indicator)

class BayesianTrainer(TrainerSubject):
    def __init__(self, model, optimizer, scheduler, loss_calculator, criterion, device, config, observer=None, train_strategy=None) -> None:

        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.LossCalculator = loss_calculator
        self.train_strategy = train_strategy
        self.preds_adapter = AdaptPreds(self.config['model_no'], self.train_strategy)
        self.model_train = model_train_register(self.train_strategy)
        self.model_predict = model_train_register(self.train_strategy)
        self.kl_loss = self.config['kl_loss']

        # bayesian config
        self.num_monte_carlo = self.config.get('num_monte_carlo', 10)
        self.train_monte_carlo = self.config.get('train_monte_carlo', 10)

        if observer is None:
            self.observer = [TimeMinor(out_after_n_batch=1000), GradMinor(out_after_n_batch=2000), LossMinor()]
        else:
            self.observer = observer

        for observer in self.observer:
            self.register_observer(observer)

        self.valid_preds_recoder = PredsRecoder()
        self.contribution_minor = ContributionMinor2()
        self.metrics = {}

    def train_step(self, data_loader):
        self.model.train()
        batch_total_time = time_tmp = time.time()
        batch_count = 0
        #for y, cont_x, cat_x, distal_x in data_loader:
        for batch in data_loader:
            batch_load_time = time.time() - time_tmp
            batch_count += 1
            sample_number = batch[0].shape[0]

            # load data
            time_tmp = time.time()

            batch = self.load_to_device(batch, self.device)
            label, inputs, sample_weight = get_inputs_labels(batch, self.train_strategy)

            # bayesian monte carlo
            output_ = []
            kl_ = []
            for mc_run in range(int(self.train_monte_carlo)):
                # preds is dict
                preds = self.model_train(inputs, self.model)
                # adapt preds to the loss calculator
                preds = self.preds_adapter.adapt(preds)
                kl = self.kl_loss(self.model)
                kl_.append(kl)
                output_.append(preds) # only check in model_no 127
            output = self._merge_mc_outputs(output_, mode="mean")
            kl = torch.mean(torch.stack(kl_), dim=0) / self.config['batch_size']

            losses = self.LossCalculator.calc_loss(output, label, self.criterion, sample_weight)
            loss = self.LossCalculator.extract_total_loss()
            loss += kl  * self.config['kl_weight']
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, error_if_nonfinite=False)
            self.notify_observers(model=self.model)
            self.optimizer.step()
            batch_train_time = time.time() - time_tmp

            # updata lr
            self.update_lr()
            batch_total_time = time.time() - batch_total_time
            # record log
            self.notify_observers(losses=losses, 
                                  sample_number = sample_number,
                                  batch_train_time=batch_train_time, 
                                  batch_load_time=batch_load_time, 
                                  batch_total_time=batch_total_time)
            batch_total_time = time_tmp = time.time()
        
        self.notify_observers(train_step_finish = True)
        self.update_lr()
        sys.stdout.flush()

    def valid_step(self, dataloader_valid):
        self.register_observer(self.valid_preds_recoder)
        self.register_observer(self.contribution_minor)
        self.model.eval()
        valid_step_time = time.time()

        pred_y_ensemble = torch.empty(0, self.config['n_class']).to(self.device)
        pred_y_uncertain = torch.empty(0, self.config['n_class']).to(self.device)

        with torch.no_grad():
            for batch in dataloader_valid:
                batch = self.load_to_device(batch, self.device)
                label, inputs, sample_weight = get_inputs_labels(batch, self.train_strategy)

                pred_results = [] # used loss calc
                output_mc = [] # used ensemble predict(mean) and uncertainty(std)
                for mc_run in range(int(self.num_monte_carlo)):
                    valid_preds = self.model_predict(inputs, self.model)
                    valid_preds = self.preds_adapter.adapt(valid_preds)
                    pred_results.append(valid_preds)
                    final_pred = self._extract_final_pred(valid_preds)
                    # ensemble, softmax first
                    output_mc.append(F.softmax(final_pred, dim=1))
                pred_results = self._merge_mc_outputs(pred_results, mode="mean")
                output_mc = torch.stack(output_mc)
                means = output_mc.mean(axis=0)
                stds = output_mc.std(axis=0)
                pred_y_ensemble = torch.cat((pred_y_ensemble, means), dim=0)
                pred_y_uncertain = torch.cat((pred_y_uncertain, stds), dim=0)


                losses = self.LossCalculator.calc_loss(pred_results, label, self.criterion, sample_weight)
                #valid_pred = self.LossCalculator.extract_pred(valid_preds)
                sample_number = batch[0].shape[0]

                self.notify_observers(losses = losses,
                                      sample_number = sample_number,
                                      valid_preds = valid_preds,
                                      label = label)
            
            self.notify_observers(valid_step_finish = True)
        valid_step_time = time.time() - valid_step_time
        logger.info("Validation used time: %.1f mins", valid_step_time / 60)
        valid_preds = self.valid_preds_recoder.output()

        self.remove_observer(self.valid_preds_recoder)
        self.remove_observer(self.contribution_minor)
        # return valid_preds
        return pred_y_ensemble, pred_y_uncertain

    def update_lr(self):
        if self.config['lr_scheduler'] != 'ROP':
            self.scheduler.step()
            if self.optimizer.param_groups[0]['lr'] < self.config['min_lr']:
                logger.debug("optimizer.param_groups[0] lr: %s", self.optimizer.param_groups[0]['lr'])
                for g in self.optimizer.param_groups:
                    g['lr'] = self.config['restart_lr']
        if self.config['lr_scheduler'] == 'ROP':
            self.scheduler.step(self.metrics['current_valid_loss'])

    def load_to_device(self, batch, device):
        if isinstance(batch, dict):
            return {k: v.to(device) for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            return [v.to(device) for v in batch]
        else:
            return batch.to(device)
    
    def _merge_mc_outputs(self, outputs, mode="mean"):
        if mode not in ["mean"]:
            raise ValueError(f"mode {mode} not supported")
        agg_fn = torch.mean
        outputs = [o[0] for o in outputs if len(o) == 2] 
        if isinstance(outputs[0], dict):
            return {k: 
                    agg_fn(torch.stack([o[k] for o in outputs]), dim=0) 
                    for k in outputs[0].keys()}, None
        elif isinstance(outputs[0], torch.Tensor):
            return agg_fn(torch.stack(outputs), dim=0), None
        
    def _extract_final_pred(self, preds):
        preds = preds[0] if len(preds) == 2 else preds
        if hasattr(preds, 'get'):
            return preds['out']
        else:
            assert isinstance(preds, torch.Tensor), "preds must be a torch.Tensor or dict-like with key 'out'"
            return preds

class Trainer(TrainerSubject):
    def __init__(self, model, optimizer, scheduler, loss_calculator, criterion, device, config, observer=None, train_strategy=None, collect_mu_r=False) -> None:

        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.LossCalculator = loss_calculator
        self.train_strategy = train_strategy
        self.preds_adapter = AdaptPreds(self.config['model_no'], self.train_strategy)
        self.model_train = model_train_register(self.train_strategy)
        self.model_predict = model_train_register(self.train_strategy)

        if observer is None:
            self.observer = [TimeMinor(out_after_n_batch=1000), GradMinor(out_after_n_batch=2000), LossMinor()]
        else:
            self.observer = observer

        for observer in self.observer:
            self.register_observer(observer)

        self.valid_preds_recoder = PredsRecoder()
        self.contribution_minor = ContributionMinor2()
        self.metrics = {}

        if collect_mu_r:
            self._mu_r_recoder = MuRRecoder()

    def train_step(self, data_loader):
        self.model.train()
        batch_total_time = time_tmp = time.time()
        batch_count = 0
        #for y, cont_x, cat_x, distal_x in data_loader:
        for batch in data_loader:
            batch_load_time = time.time() - time_tmp
            batch_count += 1
            sample_number = batch[0].shape[0]

            # load data
            time_tmp = time.time()

            batch = self.load_to_device(batch, self.device)
            label, inputs, sample_weight = get_inputs_labels(batch, self.train_strategy)
            preds = self.model_train(inputs, self.model)
            # adapt preds to the loss calculator
            preds = self.preds_adapter.adapt(preds)

            losses = self.LossCalculator.calc_loss(preds, label, self.criterion, sample_weight)
            loss = self.LossCalculator.extract_total_loss()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, error_if_nonfinite=False)
            self.notify_observers(model=self.model)
            self.optimizer.step()
            batch_train_time = time.time() - time_tmp

            # updata lr
            self.update_lr()
            batch_total_time = time.time() - batch_total_time
            # record log
            self.notify_observers(losses=losses, 
                                  sample_number = sample_number,
                                  batch_train_time=batch_train_time, 
                                  batch_load_time=batch_load_time, 
                                  batch_total_time=batch_total_time)
            batch_total_time = time_tmp = time.time()
        
        self.notify_observers(train_step_finish = True)
        self.update_lr()
        sys.stdout.flush()

    def valid_step(self, dataloader_valid):
        self.register_observer(self.valid_preds_recoder)
        self.register_observer(self.contribution_minor)
        if hasattr(self, '_mu_r_recoder'):
            self.register_observer(self._mu_r_recoder)
        self.model.eval()
        valid_step_time = time.time()
        with torch.no_grad():
            for batch in dataloader_valid:
                batch = self.load_to_device(batch, self.device)
                label, inputs, sample_weight = get_inputs_labels(batch, self.train_strategy)
                valid_preds = self.model_predict(inputs, self.model)
                valid_preds = self.preds_adapter.adapt(valid_preds)
                #label, valid_preds = model_predict(batch, self.model, self.train_strategy)
                losses = self.LossCalculator.calc_loss(valid_preds, label, self.criterion, sample_weight)
                #valid_pred = self.LossCalculator.extract_pred(valid_preds)
                sample_number = batch[0].shape[0]

                self.notify_observers(losses = losses,
                                      sample_number = sample_number,
                                      valid_preds = valid_preds,
                                      label = label)

            self.notify_observers(valid_step_finish = True)
        valid_step_time = time.time() - valid_step_time
        logger.info("Validation used time: %.1f mins", valid_step_time / 60)
        valid_preds = self.valid_preds_recoder.output()

        self.remove_observer(self.valid_preds_recoder)
        self.remove_observer(self.contribution_minor)
        if hasattr(self, '_mu_r_recoder'):
            self.remove_observer(self._mu_r_recoder)
        return valid_preds

    def get_mu_r(self):
        if hasattr(self, '_mu_r_recoder'):
            return self._mu_r_recoder.output()
        return None, None

    def update_lr(self):
        if self.config['lr_scheduler'] != 'ROP':
            self.scheduler.step()
            if self.optimizer.param_groups[0]['lr'] < self.config['min_lr']:
                logger.debug("optimizer.param_groups[0] lr: %s", self.optimizer.param_groups[0]['lr'])
                for g in self.optimizer.param_groups:
                    g['lr'] = self.config['restart_lr']
        if self.config['lr_scheduler'] == 'ROP':
            self.scheduler.step(self.metrics['current_valid_loss'])

    def load_to_device(self, batch, device):
        if isinstance(batch, dict):
            return {k: v.to(device) for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            return [v.to(device) for v in batch]
        else:
            return batch.to(device)

# to-do : same as get_inputs_labels
def model_train_register(strategy=None):
    strategy_functions = {
        'segment_soft_label': model_train_simple,
        'AvgSegmentLabel_withGAN': model_train_simple,
        'AvgSegmentLabel_withGAN2': model_train_simple,
        'segment_soft_label_step': model_train_with_step,
        'segment_soft_label_step_withGAN': model_train_with_step,
        'AvgSegMutUseInLocal': model_train_avgmut_in_local,
        'AvgSegMutAndKmerMut': model_train_with_step2,
        'AvgSegMutAndNucSkewUseInLocal': model_train_avgmut_skew_in_local,
        'AvgSegMutAndKmerMutUseInLocal': model_train_avgmut_kmer_in_local,
        'AvgStepMutAndKmerMutUseInLocal': model_train_avgmut_kmer_in_local,
        'AvgStepMutAndKmerMutCominedLoss': model_train_avgmut_kmer_in_local,
        'SKA_local': model_train_avgmut_kmer_arg_in_local,
    }
    # backward compatible
    if strategy is None:
        strategy = 'segment_soft_label'

    try:
        train_function = strategy_functions[strategy]
    except KeyError:
        raise ValueError(f"Unsupported model train strategy: '{strategy}'")
    return train_function

def model_train_simple(batch, model):
    cont_x, cat_x, distal_x = batch
    return model((cont_x, cat_x), distal_x)

def model_train_with_step(batch, model):
    cont_x, cat_x, distal_x, avg_mut_label, segment_id_label = batch
    return model((cont_x, cat_x), distal_x, avg_mut_label, segment_id_label)

def model_train_with_step2(batch, model):
    cont_x, cat_x, distal_x, avg_mut_label, kmer_mut_label = batch
    return model((cont_x, cat_x), distal_x, avg_mut_label=avg_mut_label, kmer_mut_label=kmer_mut_label)

def model_train_avgmut_in_local(batch, model):
    cont_x, cat_x, distal_x, avg_mut_label = batch
    local_input = {
        'cont_data': cont_x,
        'cat_data': cat_x,
        'avg_mutations': avg_mut_label
    }
    return model(local_input, distal_x)

def model_train_avgmut_skew_in_local(batch, model):
    cont_x, cat_x, distal_x, avg_mut_label, nuc_skew = batch
    local_input = {
        'cont_data': cont_x,
        'cat_data': cat_x,
        'avg_mutations': avg_mut_label,
        'nuc_skew': nuc_skew
    }
    return model(local_input, distal_x)

def model_train_avgmut_kmer_in_local(batch, model):
    cont_x, cat_x, distal_x, avg_mut_label, avg_kmer_mut = batch
    local_input = {
        'cont_data': cont_x,
        'cat_data': cat_x,
        'avg_mutations': avg_mut_label,
        'segment_avg_kmer_mut': avg_kmer_mut
    }
    return model(local_input, distal_x)

def model_train_avgmut_kmer_arg_in_local(batch, model):
    cont_x, cat_x, distal_x, avg_mut_label, avg_kmer_mut, arg_feature = batch
    local_input = {
        'cont_data': cont_x,
        'cat_data': cat_x,
        'avg_mutations': avg_mut_label,
        'segment_avg_kmer_mut': avg_kmer_mut,
    }
    return model(local_input, distal_x, arg_feature)

def weights_init(m):
    """Initialize network layers"""
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight)
        
        if m.bias is not None:
            #nn.init.normal_(m.bias)
            nn.init.constant_(m.bias, 0)
        
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
            
        if m.bias is not None:
            #nn.init.normal_(m.bias)
            nn.init.constant_(m.bias, 0)
        
    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(m.__getattr__(p))

class TorchBackendManager:
    def __init__(self, use_dilation=False, input_size_fixed=True):
        self.use_dilation = use_dilation
        self.input_size_fixed = input_size_fixed

    def set_torch_backends(self):
        """Configure PyTorch backend settings based on dilation usage."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = not self.use_dilation
        torch.backends.cudnn.benchmark = not self.input_size_fixed
        self.display_torch_backends_info()

    def display_torch_backends_info(self):
        """Display the current PyTorch backend settings."""
        logger.info("TF32 Matmul Enabled: %s", torch.backends.cuda.matmul.allow_tf32)
        logger.info("CUDNN Benchmark Enabled: %s", torch.backends.cudnn.benchmark)
        logger.info("CUDNN TF32 Enabled: %s", torch.backends.cudnn.allow_tf32)
        logger.info("CUDNN Deterministic: %s", torch.backends.cudnn.deterministic)

    def display_torch_device_info(self):
        """Display the current PyTorch device information."""
        logger.info("torch._C._cuda_getDeviceCount(): %s", torch._C._cuda_getDeviceCount())
        logger.info("torch.cuda.device_count(): %s", torch.cuda.device_count())

class AdaptPreds:
    """Normalise any model output into ``(PredictOutput, Optional[SegmentOutput])``.

    Input formats handled (from oldest to newest):

    * bare ``Tensor``                     → Network0 / 1 / 2
    * ``dict``                            → legacy single-dict models
    * ``(dict, None)``                    → Network3
    * ``(dict, Optional[dict])``          → MuRaL_Network3_addfc3 + variants
    * ``(Tensor, Tensor, Tensor)``        → legacy (local, distal, fused) tuple
    * ``(Tensor, Tensor, Tensor, Tensor)``→ legacy (local, mid, distal, fused) tuple
    * ``(PredictOutput, Optional[SegmentOutput])`` → already normalised (pass-through)
    """

    def __init__(self, model_no, strategy):
        self.model_no = model_no
        self.strategy = strategy

    def adapt(self, preds):
        return normalize_output(preds)


def normalize_output(preds):
    """Normalize a model's raw return to ``(PredictOutput, Optional[SegmentOutput])``."""
    from MuRaL.models.output import PredictOutput, SegmentOutput
    import torch

    # already normalised
    if isinstance(preds, tuple) and isinstance(preds[0], PredictOutput):
        return preds

    # (dict, optional_seg)  —  current standard format
    if isinstance(preds, tuple) and len(preds) == 2:
        main, seg = preds
        po = _dict_to_predict_output(main)
        if seg is not None and isinstance(seg, dict):
            so = SegmentOutput(**{k: v for k, v in seg.items() if hasattr(SegmentOutput, k)})
        elif isinstance(seg, SegmentOutput):
            so = seg
        else:
            so = None
        return po, so

    # legacy positional tuples: (local, distal, fused) or (local, mid, distal, fused)
    if isinstance(preds, tuple) and len(preds) >= 3:
        po = PredictOutput()
        if len(preds) == 3:
            po.local, po.distal, po.out = preds
        elif len(preds) >= 4:
            po.local, po.mid, po.distal, po.out = preds[:4]
        return po, None

    # dict  —  legacy single-dict output
    if isinstance(preds, dict):
        return _dict_to_predict_output(preds), None

    # bare Tensor  —  Network0 / 1 / 2
    if isinstance(preds, torch.Tensor):
        return PredictOutput(out=preds), None

    return preds


def _dict_to_predict_output(d):
    """Copy known keys from *d* into a new PredictOutput."""
    from MuRaL.models.output import PredictOutput
    po = PredictOutput()
    for k, v in d.items():
        if hasattr(po, k):
            setattr(po, k, v)
    return po

@dataclass
class BatchConfig:
    has_avg_mut: bool = False
    has_kmer_mut: bool = False
    has_arg_feature: bool = False

    include_avg_mut_in_inputs: bool = False
    include_kmer_mut_in_inputs: bool = False
    inclued_arg_feature_in_inputs: bool = False

    include_avg_mut_in_labels: bool = False
    include_kmer_mut_in_labels: bool = False
    include_arg_feature_in_labels: bool = False

# 策略注册表
# 策略命名规范：
# [特征]_[标签类型]_[输入模式]
# 
# 特征代号:
#   S = Segment mutation rate
#   K = Kmer mutation rate  
#   A = ARG features
#   N = Nucleotide skew
#
# 标签类型:
#   hard = 硬标签
#   soft = 软标签 (segment level)
#   step = 软标签 (step level)
#
# 输入模式:
#   loss = 仅用于loss计算
#   local = 作为local模型输入
#   gan = 使用GAN
STRATEGY_CONFIGS: Dict[str, BatchConfig] = {
    'AvgStepMutAndKmerMutUseInLocal': BatchConfig(
        has_avg_mut=True, has_kmer_mut=True,
        include_avg_mut_in_inputs=True, include_kmer_mut_in_inputs=True
    ),

    'SKA_local': BatchConfig(
        has_avg_mut=True, has_kmer_mut=True, has_arg_feature=True,
        include_avg_mut_in_inputs=True, include_kmer_mut_in_inputs=True, inclued_arg_feature_in_inputs=True
    ),

}

def get_inputs_labels(batch, strategy=None):
    """统一的batch处理函数"""

    # 默认策略：兼容 dict_to_tuple_collate 格式 (y, cat_x, distal_x, ...)
    if strategy is None:
        batch_iter = iter(batch)
        y = next(batch_iter)
        cat_x = next(batch_iter)
        distal_x = next(batch_iter)
        sample_weight = None
        try:
            sample_weight = next(batch_iter)
        except StopIteration:
            pass
        return {'label': _process_label(y)}, (0, cat_x, distal_x), sample_weight

    config = STRATEGY_CONFIGS.get(strategy)
    if config is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 解包batch（根据配置）
    batch_iter = iter(batch)
    y = next(batch_iter)
    cat_x = next(batch_iter)
    distal_x = next(batch_iter)

    segment_avg_mut = next(batch_iter) if config.has_avg_mut else None
    kmer_mut = next(batch_iter) if config.has_kmer_mut else None
    arg_feature = next(batch_iter) if config.has_arg_feature else None

    # 尝试获取 sample_weight
    sample_weight = None
    try:
        sample_weight = next(batch_iter)
    except StopIteration:
        pass

    # 构建labels
    labels = {'label': _process_label(y)}
    if config.include_avg_mut_in_labels and segment_avg_mut is not None:
        labels['avg_mut'] = segment_avg_mut
    if config.include_kmer_mut_in_labels and kmer_mut is not None:
        labels['avg_kmer_mut'] = kmer_mut

    # 构建inputs
    inputs = [0, cat_x, distal_x]  # cont_x = 0
    if config.include_avg_mut_in_inputs:
        inputs.append(segment_avg_mut)
    if config.include_kmer_mut_in_inputs:
        inputs.append(kmer_mut)
    if config.inclued_arg_feature_in_inputs:
        inputs.append(arg_feature)

    return labels, tuple(inputs), sample_weight

def _process_label(y):
    """统一的label处理"""
    if y.dim() > 1 and y.shape[1] == 1:
        return y.long().squeeze(1)
    return y.long()
