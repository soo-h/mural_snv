import time
import numpy as np
import sys
import torch
from MuRaL.evaluation.evaluation import print_gradients, print_gradient_norms

from typing import Dict, Any, Union



class Observer:
    def update(self, **kwargs):
        raise NotImplementedError


class TimeMinor(Observer):
    def __init__(self, out_after_n_batch, dataset_clss='Training', printer=print):
        self.batch_load_times = []
        self.batch_train_times = []
        self.batch_total_times = []
        self.conter = 0
        self.out_after_n_batch = out_after_n_batch
        self.dataset_clss = dataset_clss
        self.printer = printer
    
    def record_batch_load(self, time):
        self.batch_load_times.append(time)

    def record_batch_train_times(self, time):
        self.batch_train_times.append(time)

    def record_batch_total_time(self, time):
        self.batch_total_times.append(time)

    def out_batch_times(self):
        self.printer(f"{self.dataset_clss} load {self.out_after_n_batch} batch used {np.sum(self.batch_load_times) / 60} min")
        self.printer(f"{self.dataset_clss} train {self.out_after_n_batch} batch used {np.sum(self.batch_train_times) / 60} min")
        self.printer(f"{self.dataset_clss} after {self.out_after_n_batch} batch used {np.sum(self.batch_total_times) / 60} min")
        self.reset()

    def reset(self):
        self.conter = 0
        self.batch_load_times = []
        self.batch_train_times = []
        self.batch_total_times = []

    def update(self, **kwargs):
        counter = False
        if 'batch_load_time' in kwargs:
            self.record_batch_load(kwargs['batch_load_time'])
            counter = True

        if 'batch_train_time' in kwargs:
            self.record_batch_train_times(kwargs['batch_train_time'])
            counter = True
            
        if 'batch_total_time' in kwargs: 
            self.record_batch_total_time(kwargs['batch_total_time'])
            counter = True

        if counter:
            self.conter += 1
            if self.conter == self.out_after_n_batch:
                self.out_batch_times()

class GradMinor(Observer):
    def __init__(self, out_after_n_batch, first_epoch=True, printer=print):
        self.grad_norms = []
        self.counter = 0
        self.out_after_n_batch = out_after_n_batch
        if first_epoch:
            self.out_epoch = 5
        else:
            self.out_epoch = 0
        self.printer = printer
    
    def out_grad(self, model):
        print_gradients(model, print=self.printer)
        print_gradient_norms(model, print=self.printer)
        self.reset()

    def reset(self):
        self.counter = 0

    def update(self, **kwargs):
        if 'model' in kwargs:
            self.counter += 1
            if self.out_epoch:
                self.out_grad(kwargs['model'])
                self.out_epoch -= 1

            if self.counter == self.out_after_n_batch:
                self.out_grad(kwargs['model'])
        
class LossMinorStrategy:
    def __init__(self):
        self.loss = []

    def record(self, loss):
        raise NotImplementedError("Must implement record method.")

    def reset(self):
        raise NotImplementedError("Must implement record method.")

    def get_total_loss(self):
        raise NotImplementedError("Must implement record method.")

class OnlyCombinedLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.printer = printer

    def record(self, loss):
        self.loss.append(loss.item())
    
    def reset(self):
        self.loss.clear()
    
    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Total Loss: {np.sum(self.loss) / sample_number}")
        return {'loss' : self.loss}
    
class LocalDistalCombinedLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.local_loss = []
        self.distal_loss = []
        self.printer = printer 

    def record(self, losses):
        if len(losses) != 3:
            raise ValueError("Expected losses to be a tuple of length 3.")
        self.local_loss.append(losses[0].item())
        self.distal_loss.append(losses[1].item())
        self.loss.append(losses[2].item())  
    
    def reset(self):
        self.local_loss.clear()
        self.distal_loss.clear()
        self.loss.clear()
    
    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}")
        self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}")
        self.printer(f"{dataset_class} Total Loss : {np.sum(self.loss) / sample_number}")
        return {'loss' : self.loss}

    def get_total_loss(self):
        return super().get_total_loss()

class LocalMidDistalCombinedLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.local_loss = []
        self.distal_loss = []
        self.mid_loss = []
        self.printer = printer

    def record(self, loss):
        self.local_loss.append(loss[0].item())
        self.mid_loss.append(loss[1].item())
        self.distal_loss.append(loss[2].item())
        self.loss.append(loss[3].item())

    def reset(self):
        self.loss.clear()
        self.local_loss.clear()
        self.distal_loss.clear()
        self.mid_loss.clear()
    
    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}")
        self.printer(f"{dataset_class} Mid Loss: {np.sum(self.mid_loss) / sample_number}")
        self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}")
        self.printer(f"{dataset_class} Total Loss : {np.sum(self.loss) / sample_number}")
        return {'loss' : self.loss}

class LocalMidDistalCombinedDecoderLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.local_loss = []
        self.distal_loss = []
        self.mid_loss = []
        self.decoder_loss = []
        self.printer = printer

    def record(self, loss):
        self.local_loss.append(loss[0].item())
        self.mid_loss.append(loss[1].item())
        self.distal_loss.append(loss[2].item())
        self.loss.append(loss[3].item())
        self.decoder_loss.append(loss[4].item())
    
    def reset(self):
        self.local_loss.clear()
        self.mid_loss.clear()
        self.distal_loss.clear()
        self.decoder_loss.clear()
        self.loss.clear()
        
    
    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}")
        self.printer(f"{dataset_class} Mid Loss: {np.sum(self.mid_loss) / sample_number}")
        self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}")
        self.printer(f"{dataset_class} construct Loss : {np.sum(self.decoder_loss) / sample_number}")
        self.printer(f"{dataset_class} Total Loss : {np.sum(self.loss) / sample_number}")
        return {'loss' : self.loss}

class AdaptiveLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print) -> None:
        super().__init__()
        self._record_init()
        self.printer = printer
    
    def _record_init(self):
        self.loss = []
        self.local_loss = []
        self.distal_loss = []
        self.mid_loss = []
        self.decoder_loss = []
 
    
    def record(self, loss):

        if isinstance(loss, torch.Tensor):
            self.loss.append(loss.item())
        elif isinstance(loss, tuple):
            if len(loss) == 3:
                self.local_loss.append(loss[0].item())
                self.distal_loss.append(loss[1].item())
                self.loss.append(loss[2].item())
            elif len(loss) == 4:
                self.local_loss.append(loss[0].item())
                self.mid_loss.append(loss[1].item())
                self.distal_loss.append(loss[2].item())
                self.loss.append(loss[3].item())
            elif len(loss) == 5:
                self.local_loss.append(loss[0].item())
                self.mid_loss.append(loss[1].item())
                self.distal_loss.append(loss[2].item())
                self.loss.append(loss[3].item())
                self.decoder_loss.append(loss[4].item())
            else:
                self.printer("Warming: Loss Record unnormal")
    def reset(self):
        self._record_init()

    def out_mean_loss(self, dataset_class, sample_number):
        if self.local_loss:
            self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}")
        if self.mid_loss:
            self.printer(f"{dataset_class} Mid Loss: {np.sum(self.mid_loss) / sample_number}")
        if self.distal_loss:
            self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}")
        if self.decoder_loss:
            self.printer(f"{dataset_class} construct Loss : {np.sum(self.decoder_loss) / sample_number}")
        if self.loss:
            loss = np.sum(self.loss) / sample_number
            self.printer(f"{dataset_class} Total Loss : {loss}") 
        return {'loss' : loss}

class AdaptiveLossStrategyLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.printer = printer
        # self._record_init()
        self.record_init = None

    def reset(self):
        self.record_init = None
    
    def _record_init(self, loss):
        # to-del, 2026-1-5 save it for back compatibility
        if not isinstance(loss, dict):
            self.keys = ['loss', 'local_loss', 'mid_loss', 'distal_loss', 'local2_loss', 'local3_loss', 'avg_mut_loss_total']
        else:
            self.keys = list(loss.keys())
        # for key in ['loss', 'local_loss', 'mid_loss', 'distal_loss', 'local2_loss', 'local3_loss', 'avg_mut_loss_total']:
        for key in self.keys:
            setattr(self, key, [])

        self.record_init = True

        """ 
        self.loss = []
        self.local_loss = []
        self.local2_loss = []
        self.local3_loss = []
        self.mid_loss = []
        self.distal_loss = []
        """

    def record(self, loss):
        if self.record_init is None:
            self._record_init(loss)
        
        for key in self.keys:
            if loss.get(key):
                getattr(self, key).append(loss[key].item())

        # for key in ['loss']:
        #     getattr(self, key).append(loss[key].item())
        # for key in ['local2_loss', 'local3_loss', 'distal_loss', 'local_loss', 'mid_loss', 'avg_mut_loss_total']:
        #     if loss.get(key):
        #         getattr(self, key).append(loss[key].item())

    def out_mean_loss(self, dataset_class, sample_number):
        for key in self.keys:
            self.printer(f"{dataset_class} {key}: {np.sum(getattr(self, key)) / sample_number}; Batch Var: {np.var(getattr(self, key))}")

        # if self.local_loss:
        #     self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}; Batch Var: {np.var(self.local_loss)}")
        # if self.local2_loss:
        #     self.printer(f"{dataset_class} Local2 Loss: {np.sum(self.local2_loss) / sample_number}; Batch Var: {np.var(self.local2_loss)}")
        # if self.local3_loss:
        #     self.printer(f"{dataset_class} Local3 Loss: {np.sum(self.local3_loss) / sample_number}; Batch Var: {np.var(self.local3_loss)}")
        # if self.mid_loss:
        #     self.printer(f"{dataset_class} Mid Loss: {np.sum(self.mid_loss) / sample_number}; Batch Var: {np.var(self.mid_loss)}")
        # if self.distal_loss:
        #     self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}; Batch Var: {np.var(self.distal_loss)}")
        # if self.avg_mut_loss_total:
        #     self.printer(f"{dataset_class} Avg Mut Loss: {np.sum(self.avg_mut_loss_total) / sample_number}; Batch Var: {np.var(self.avg_mut_loss_total)}")
        # self.printer(f"{dataset_class} Total Loss(Mix Loss) : {np.sum(self.loss) / sample_number} ; Batch Var: {np.var(self.loss)}")

        return {'loss' : np.sum(self.loss) / sample_number}
    
class SegmentCombinedLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.printer = printer
        self._record_init()

    def reset(self):
        self._record_init()
    
    def _record_init(self):
        self.loss = []
        self.local_loss = []
        self.mid_loss = []
        self.distal_loss = []
        self.segment_id_loss = []
        self.avg_mut_loss = []
        self.avg_kmer_mut_loss = []

        self.total_loss = []
        self.discrim_loss = []
        self.construct_loss = []

    def record(self, loss):
        self.loss.append(loss['loss'].item())
        self.local_loss.append(loss['local_loss'].item())
        self.mid_loss.append(loss['mid_loss'].item())
        self.distal_loss.append(loss['distal_loss'].item())
        if loss.get('segment_id_loss'):
            self.segment_id_loss.append(loss['segment_id_loss'].item())
        self.avg_mut_loss.append(loss['avg_mut_loss'].item())
        # not torch
        if loss.get('total_loss'):
            self.total_loss.append(loss['total_loss'].item())
        if loss.get('avg_kmer_mut_loss'):
            self.avg_kmer_mut_loss.append(loss['avg_kmer_mut_loss'].item())
        if loss.get('discrim_loss'):
            self.discrim_loss.append(loss['discrim_loss'])
        # torch
        if loss.get('construct_loss'):
            self.construct_loss.append(loss['construct_loss'].item())



    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}")
        self.printer(f"{dataset_class} Mid Loss: {np.sum(self.mid_loss) / sample_number}")
        self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}")
        self.printer(f"{dataset_class} Avg Mut Loss: {np.sum(self.avg_mut_loss) / sample_number}")
        self.printer(f"{dataset_class} Total Loss(Main Loss) : {np.sum(self.loss) / sample_number}")
        if self.segment_id_loss:
            self.printer(f"{dataset_class} Segment Id Loss : {np.sum(self.segment_id_loss) / sample_number}")
        if self.avg_kmer_mut_loss:
            self.printer(f"{dataset_class} Avg Kmer Mut Loss : {np.sum(self.avg_kmer_mut_loss) / sample_number}")
        if self.total_loss:
            self.printer(f"{dataset_class} Total Loss(Combined Loss) : {np.sum(self.total_loss) / sample_number}")
        if self.discrim_loss:
            self.printer(f"{dataset_class} Discrim Loss: {np.sum(self.discrim_loss) / sample_number}; Batch Var: {np.var(self.discrim_loss)}")
        if self.construct_loss:
            self.printer(f"{dataset_class} Construct Loss (mean): {np.sum(self.construct_loss) / sample_number}; Batch Var: {np.var(self.construct_loss)}")

        return {'loss' : np.sum(self.loss) / sample_number}

class SoftLabelUtilAvgSegmentWithGANMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.printer = printer
        self._record_init()

    def reset(self):
        self._record_init()
    
    def _record_init(self):
        self.loss = []
        self.local_loss = []
        self.mid_loss = []
        self.distal_loss = []

        self.construct_loss = []
        
        self.hard_label_loss = []
        self.soft_label_loss = []
        self.mix_loss = []

    def record(self, loss):
        self.loss.append(loss['loss'].item())
        self.local_loss.append(loss['local_loss'].item())
        self.mid_loss.append(loss['mid_loss'].item())
        self.distal_loss.append(loss['distal_loss'].item())

        # torch
        if loss.get('construct_loss') is not None:
            self.construct_loss.append(loss['construct_loss'].item())

        if loss.get('soft_label_loss') is not None:
            self.soft_label_loss.append(loss['soft_label_loss'].item())

        if loss.get('mix_loss') is not None:
            self.mix_loss.append(loss['mix_loss'].item())
        
        self.hard_label_loss.append(loss['hard_label_loss'].item())



    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}; Batch Var: {np.var(self.local_loss)}")
        self.printer(f"{dataset_class} Mid Loss: {np.sum(self.mid_loss) / sample_number}; Batch Var: {np.var(self.mid_loss)}")
        self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}; Batch Var: {np.var(self.distal_loss)}")
        self.printer(f"{dataset_class} Total Loss(Mix Loss) : {np.sum(self.loss) / sample_number} ; Batch Var: {np.var(self.loss)}")

        if self.construct_loss:
            self.printer(f"{dataset_class} Construct Loss (mean): {np.sum(self.construct_loss) / sample_number}; Batch Var: {np.var(self.construct_loss)}")

        if self.soft_label_loss:
            self.printer(f"{dataset_class} Soft Loss (mean): {np.sum(self.soft_label_loss) / sample_number}; Batch Var: {np.var(self.construct_loss)}")

        self.printer(f"{dataset_class} Hard Label Loss: {np.sum(self.hard_label_loss) / sample_number}; Batch Var: {np.var(self.hard_label_loss)}")

        if self.mix_loss:
            self.printer(f"{dataset_class} Mix Loss (mean): {np.sum(self.mix_loss) / sample_number}; Batch Var: {np.var(self.construct_loss)}")

        return {'loss' : np.sum(self.loss) / sample_number}


class LossMinor(Observer):
    strategy_map = {
        'OnlyCombined': OnlyCombinedLossMinorStrategy,
        'LocalDistalCombined': LocalDistalCombinedLossMinorStrategy,
        'LocalMidDistalCombined': LocalMidDistalCombinedLossMinorStrategy,
        'LocalMidDistalCombinedDecoder': LocalMidDistalCombinedDecoderLossMinorStrategy,  # Add as needed
        'segment_soft_label' : SegmentCombinedLossMinorStrategy,
        'segment_soft_label_step' : SegmentCombinedLossMinorStrategy,
        'segment_soft_label_step_withGAN' : SegmentCombinedLossMinorStrategy,

        'AvgSegmentLabel_withGAN' : SoftLabelUtilAvgSegmentWithGANMinorStrategy,
        'AvgSegmentLabel_withGAN2' : SoftLabelUtilAvgSegmentWithGANMinorStrategy,

        'AvgSegMutUseInLocal' : AdaptiveLossStrategyLossMinorStrategy,
        'AvgSegMutAndKmerMut' : SegmentCombinedLossMinorStrategy,
        'AvgSegMutAndNucSkewUseInLocal' : AdaptiveLossStrategyLossMinorStrategy,
        'AvgSegMutAndKmerMutUseInLocal': AdaptiveLossStrategyLossMinorStrategy,
        'AvgStepMutAndKmerMutUseInLocal': AdaptiveLossStrategyLossMinorStrategy,
        'AvgStepMutAndKmerMutCominedLoss': AdaptiveLossStrategyLossMinorStrategy,
        'SKA_local' : AdaptiveLossStrategyLossMinorStrategy,
    }

    def __init__(self, calc_loss_strategy_name=None, printer=print):
        if calc_loss_strategy_name is None:
            self.loss_strategy = AdaptiveLossStrategyLossMinorStrategy()

        else:
            if calc_loss_strategy_name not in self.strategy_map:
                sys.exit(f"Error: Unsupported strategy name not in {self.strategy_map}")
            self.loss_strategy = self.strategy_map[calc_loss_strategy_name]()
        self.printer = printer
        self.sample_number = 0

    def record_loss(self, loss):
        self.loss_strategy.record(loss)
    
    def record_sample_number(self, sample_number):
        self.sample_number += sample_number

    def reset(self):
        self.loss_strategy.reset()
        self.sample_number = 0

    def out_mean_losses(self, dataset_class, sample_number):
        minor_dict = self.loss_strategy.out_mean_loss(dataset_class, sample_number)
        self.reset()
        return minor_dict

    def update(self, **kwargs):
        if 'losses' in kwargs:
            self.record_loss(kwargs['losses'])
            self.record_sample_number(kwargs['sample_number'])
        if 'train_step_finish' in kwargs:
            minor_dict = self.out_mean_losses(dataset_class="Training", sample_number=self.sample_number)
            return {}

        if 'valid_step_finish' in kwargs:
            minor_dict = self.out_mean_losses(dataset_class="Validation", sample_number=self.sample_number)
            return {'valid_loss': minor_dict['loss']}

        

class ModelSaverObserve(Observer):
    def __init__(self, model_saver, printer=print):
        self.model_saver = model_saver
        self.printer = printer

    def update(self, **kwargs):
        if 'epoch_finish' in kwargs:
            self.model_saver.save_model(kwargs['epoch_finish'])
            self.printer(f"Model saved at epoch {kwargs['epoch_finish']}")

class PredsRecoder(Observer):

    def __init__(self) -> None:
        super().__init__()

        self.preds = None
    
    def reset(self):
        self.preds = None
    
    def recode(self, preds):
        preds = self.extract_preds(preds)

        if self.preds is None:
            self.preds = preds
        else:
            self.preds = torch.cat([self.preds, preds], dim=0)
        
    def extract_preds(self, preds):
        # Segment soft label strategy
        if isinstance(preds, dict):
            return preds['out']
        elif isinstance(preds[0], dict):
            predict_out = preds[0]
            pred = predict_out['out']
        else:
            if isinstance(preds, tuple):
                if len(preds) == 3:
                    preds_local, preds_distal, pred = preds
                if len(preds) == 4:
                    preds_local, preds_mid, preds_distal, pred = preds
                if len(preds) == 5:
                    preds_local, preds_mid, preds_distal, pred, loss_construct = preds


        return pred

    
    def output(self):
        preds = self.preds
        self.reset()
        return preds

    
    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            preds = kwargs['valid_preds']
            self.recode(preds)


class MuRRecoder(Observer):
    """收集 predict_out 中的 mu 和 r（已激活的 NB 参数），供 validation 评估使用。

    通过 Trainer 的 collect_mu_r flag 内部注册，生命周期与 valid_preds_recoder 一致。
    调用方通过 Trainer.get_mu_r() 获取结果。
    """

    def __init__(self) -> None:
        super().__init__()
        self.mu = None
        self.r = None

    def reset(self):
        self.mu = None
        self.r = None

    def recode(self, preds):
        predict_out, _ = preds if isinstance(preds, tuple) else (preds, None)
        mu = predict_out.get('mu')
        r = predict_out.get('r')
        if mu is None:
            return  # CE 模型没有 mu/r，静默跳过
        batch_size = mu.shape[0]
        self.mu = mu if self.mu is None else torch.cat([self.mu, mu], dim=0)
        self.r = r if self.r is None else torch.cat([self.r, r], dim=0)

    def output(self):
        mu, r = self.mu, self.r
        self.reset()
        return mu, r

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            self.recode(kwargs['valid_preds'])


class SubModelPredResRecoder(Observer):
    def __init__(self):
        # 使用字典统一管理模型结果，便于扩展
        self.model_results = {
            'local': None,
            'local2': None,
            'local3': None,
            'mid': None,
            'distal': None,
            'out': None
        }

    # ===== 主功能方法 =====
    def update(self, **kwargs):
        """Observer接口实现, 响应模型训练或验证事件"""
        if 'valid_preds' in kwargs:
            preds_each_model = self._extract_sub_model_preds(kwargs['valid_preds'])
            if preds_each_model:
                self._record(preds_each_model)

    def output(self):
        """输出并重置记录的结果"""
        res_dict = {key: val for key, val in self.model_results.items() if val is not None}
        self.reset()
        return res_dict

    def reset(self):
        """重置记录"""
        self.model_results = {key: None for key in self.model_results}

    # ===== 核心逻辑 =====
    def _record(self, preds_each_model: Dict[str, torch.Tensor]):
        """记录每轮验证子模型的预测"""
        for model_name, preds in preds_each_model.items():
            if model_name not in self.model_results:
                raise ValueError(f"Unsupported model name: {model_name}")
            if self.model_results[model_name] is None:
                self.model_results[model_name] = preds
            else:
                self.model_results[model_name] = torch.cat([self.model_results[model_name], preds], dim=0)

    # ===== 数据提取方法 =====
    def _extract_sub_model_preds(self, preds: Union[Dict, tuple]) -> Dict[str, torch.Tensor]:
        """提取子模型预测"""
        if isinstance(preds[0], dict):
            return {k: v for k, v in preds[0].items() if v is not None}
        elif isinstance(preds, tuple):
            return self._extract_from_tuple(preds)
        raise ValueError(f"Unsupported format for predictions: {type(preds)}")

    def _extract_from_tuple(self, preds: tuple) -> Dict[str, torch.Tensor]:
        """处理元组格式的子模型预测"""
        length_map = {
            3: ('local', 'distal', 'out'),
            4: ('local', 'mid', 'distal', 'out'),
            5: ('local', 'mid', 'distal', 'out', '_')
        }
        if len(preds) not in length_map:
            raise ValueError(f"Unsupported tuple format for sub-model predictions: {len(preds)}")
        return {name: preds[i] for i, name in enumerate(length_map[len(preds)]) if name != '_'}



class ContributionMinor(Observer):
    def __init__(self, printer=print):
        self.mean_contributions = None
        self.calc_var_contributions = None
        self.printer = printer

    def record(self, preds_each_model: dict):

        mean_contributions = self.calc_mean_contribution(preds_each_model)
        var_contributions = self.calc_var_contribution(preds_each_model)

        if self.mean_contributions is None:
            self.mean_contributions = mean_contributions
            self.var_contributions = var_contributions
        else:
            self.update_contributions('mean_contribution', mean_contributions)
            self.update_contributions('var_contribution', var_contributions)
    
    def calc_mean_contribution(self, preds_each_model: dict):
        mean_contributions = {}
        fused_preds = self.convert_to_numpy(preds_each_model['fused_pred'])
        # Note: each model output should after softmax or sigmoid to ensure each element large than 0
        for model_name, preds in preds_each_model.items():
            if model_name == 'fused_pred':
                continue
            preds = self.convert_to_numpy(preds)
            #contribution = np.mean(preds / fused_preds, axis=0)
            contribution = np.mean(preds, axis=0)
            mean_contributions[model_name] = [contribution]
        return mean_contributions

    def calc_var_contribution(self, preds_each_model: dict):
        var_contributions = {}

        for model_name, preds in preds_each_model.items():
            if model_name == 'fused_pred':
                continue
            preds = self.convert_to_numpy(preds)
            contribution = np.var(preds, axis=0)
            var_contributions[model_name] = [contribution]
        return var_contributions 
    
    def update_contributions(self, contribution_name , contributions):

        if contribution_name == 'mean_contribution':
            for model_name, contribution in contributions.items():
                self.mean_contributions[model_name] = np.concatenate([self.mean_contributions[model_name], contributions[model_name]])

        elif contribution_name == 'var_contribution':
            for model_name, contribution in contributions.items():
                self.var_contributions[model_name] = np.concatenate([self.var_contributions[model_name], contributions[model_name]])
        else:
            raise ValueError("Error: For ContributionMinor, contribution_name must be 'mean_contribution' or 'var_contribution'")
    
    def convert_to_numpy(self, contribution):
        if contribution.is_cuda:
            contribution = contribution.cpu().numpy()
        else:
            contribution = contribution.numpy()
        return contribution
    
    def extract_sub_model_preds(self, preds):
        if isinstance(preds[0], dict):
            preds = preds[0]
            if preds.get('local2') is not None:
                if preds.get('local3') is None:
                    return {
                        'local': preds['local'],
                        'local2': preds['local2'],
                        'mid': preds['mid'],
                        'distal': preds['distal'],
                        'fused_pred': preds['out']
                    }
                else:
                    return {
                        'local': preds['local'],
                        'local2': preds['local2'],
                        'local3': preds['local3'],
                        'mid': preds['mid'],
                        'distal': preds['distal'],
                        'fused_pred': preds['out']
                    }
            
            return {
                'local': preds['local'],
                'mid': preds['mid'],
                'distal': preds['distal'],
                'fused_pred': preds['out']
            }
 
        
        if isinstance(preds, tuple):
            if len(preds) == 3:
                preds_local, preds_distal, pred = preds
                return {
                    'local': preds_local,
                    'distal': preds_distal,
                    'fused_pred': pred
                }
            if len(preds) == 4:
                preds_local, preds_mid, preds_distal, pred = preds

            if len(preds) == 5:
                preds_local, preds_mid, preds_distal, pred, loss_construct = preds

            return {
                    'local': preds_local,
                    'mid': preds_mid,
                    'distal': preds_distal,
                    'fused_pred': pred
                }

        return None

    def reset(self):
        self.mean_contributions = None
        self.var_contributions = None
    
    def out_contribution(self):
        if self.mean_contributions is None:
            self.printer("Only one preds out, No contribution to report.")
            return

        for model_name, contribution in self.mean_contributions.items():
            self.printer(f"{model_name} Mean abs contribution in validation:", np.mean(contribution, axis=0))

        for model_name, contribution in self.var_contributions.items():
            self.printer(f"{model_name} Mean Var in validation:", np.mean(contribution, axis=0))

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            preds = kwargs['valid_preds']
            preds_each_model = self.extract_sub_model_preds(preds)
            if preds_each_model is not None:
                self.record(preds_each_model)

        if 'valid_step_finish' in kwargs:
            self.out_contribution()
            self.reset()

class ContributionMinor2_(Observer):
    """
    1. optimi according to ContributionMinor
    2. change contribution save method
        |--- record contribution for each mut type
        |--- fix bug for Var contribution
    """

    def __init__(self, printer=print):
        self.mean_contributions = None
        self.var_contributions = None
        self.printer = printer

    # ===== 主功能方法 =====
    def update(self, **kwargs):
        """Observer接口实现, 响应模型训练或验证事件"""
        if 'valid_preds' in kwargs:
            mut_labels = self.extract_mut_labels(kwargs['label'])
            mut_labels = mut_labels.view(-1)
            preds_each_model = self.extract_sub_model_preds(kwargs['valid_preds'])
            if preds_each_model:
                self.record(preds_each_model, mut_labels)

        if 'valid_step_finish' in kwargs:
            self.report_contributions()
            self.reset()

    # ===== 核心逻辑 =====
    def record(self, preds_each_model: dict, mut_labels: torch.Tensor):
        """记录每轮验证子模型贡献"""
        mean_contributions = self._calc_contribution(preds_each_model, mut_labels, method="mean")
        var_contributions = self._calc_contribution(preds_each_model, mut_labels, method="var")

        # 初始化或更新贡献数据
        if self.mean_contributions is None:
            self.mean_contributions = mean_contributions
            self.var_contributions = var_contributions
        else:
            self._update_contributions(self.mean_contributions, mean_contributions)
            self._update_contributions(self.var_contributions, var_contributions)

    def report_contributions(self):
        """输出每个子模型的贡献信息"""
        if self.mean_contributions is None:
            self.printer("No contributions recorded yet.")
            return

        self._print_contribution("Mean absolute contribution", self.mean_contributions)
        self._print_contribution("Variance of contribution", self.var_contributions)

    def reset(self):
        """重置贡献数据"""
        self.mean_contributions = None
        self.var_contributions = None

    # ===== 私有工具方法 =====
    def _calc_contribution(self, preds_each_model: dict, mut_labels:torch.tensor , method: str):
        """计算子模型的贡献，支持均值或方差"""
        contributions = {}
        mut_labels = self._convert_to_numpy(mut_labels)

        for model_name, preds in preds_each_model.items():
            if model_name == 'fused_pred':
                continue
        
            preds = self._convert_to_numpy(preds)
        
            # 获取类别数和初始化贡献矩阵
            num_classes = 4
            contribution_matrix = np.zeros((num_classes, preds.shape[1]))
        
            for label in range(num_classes):
                # 获取当前类别的预测
                class_preds = preds[mut_labels == label]
                if class_preds.shape[0] > 0:  # 如果存在该类别的数据
                    if method == "mean":
                        contribution_matrix[label] = np.mean(class_preds, axis=0)
                    elif method == "var":
                        contribution_matrix[label] = np.var(class_preds, axis=0)
                    else:
                        raise ValueError("Unsupported method for contribution calculation.")

            contributions[model_name] = [contribution_matrix]
    
        return contributions

    def _calc_batch_stats(self, preds, mut_labels, num_classes):
        """计算当前批次的均值、方差和样本计数"""
        preds = self._convert_to_numpy(preds)
        mut_labels = self._convert_to_numpy(mut_labels)
        num_features = preds.shape[1]

        batch_mean = np.zeros((num_classes, num_features))
        batch_var = np.zeros((num_classes, num_features))
        batch_counts = np.zeros(num_classes)

        for label in range(num_classes):
            class_preds = preds[mut_labels == label]
            if class_preds.shape[0] > 0:
                batch_mean[label] = np.mean(class_preds, axis=0)
                batch_var[label] = np.var(class_preds, axis=0)
                batch_counts[label] = class_preds.shape[0]

        return batch_mean, batch_var, batch_counts

    def _update_contributions(self, existing, new):
        """更新现有的贡献数据"""
        for model_name, contribution in new.items():
            existing[model_name] = np.concatenate([existing[model_name], contribution],axis=0)

    def _print_contribution(self, title, contributions):
        """格式化输出贡献数据"""
        self.printer(f"{title}:")
        for model_name, contribution in contributions.items():
            self.printer(f"  {model_name}: {np.mean(contribution, axis=0)}")

    def _convert_to_numpy(self, tensor):
        """将张量转换为numpy数组"""
        if hasattr(tensor, 'is_cuda') and tensor.is_cuda:
            return tensor.cpu().numpy()
        return tensor.numpy()

    # ===== 数据提取方法 =====
    def extract_mut_labels(self, label):
        """从标签数据中提取突变标签"""
        if isinstance(label, dict):
            return label['label']
        return label
    def extract_sub_model_preds(self, preds):
        """根据数据格式提取子模型预测"""
        if isinstance(preds[0], dict):
            return {k: v for k, v in preds[0].items() if v is not None}
        if isinstance(preds, tuple):
            return self._extract_from_tuple(preds)
        return None

    def _extract_from_dict(self, preds):
        """处理字典格式的子模型预测"""
        keys = ['local', 'local2', 'local3', 'mid', 'distal', 'out']
        available_keys = {k: preds[k] for k in keys if preds[k] is not None}
        available_keys['fused_pred'] = available_keys.pop('out', None)
        return available_keys

    def _extract_from_tuple(self, preds):
        """处理元组格式的子模型预测"""
        if len(preds) == 3:
            preds_local, preds_distal, pred = preds
            return {'local': preds_local, 'distal': preds_distal, 'fused_pred': pred}
        if len(preds) == 4:
            preds_local, preds_mid, preds_distal, pred = preds
        elif len(preds) == 5:
            preds_local, preds_mid, preds_distal, pred, _ = preds
        else:
            raise ValueError("Unsupported tuple format for sub-model predictions.")
        return {'local': preds_local, 'mid': preds_mid, 'distal': preds_distal, 'fused_pred': pred}

# class ContributionMinor2(Observer):
#     """
#     优化代码以支持批次累积计算贡献。
#     使用分批均值和方差计算整体方差的方式。
#     """

#     def __init__(self, printer=print):
#         self.prob0_sumary_stats = {
#             'sample_number' : 0,
#             'Mean': {},
#             'Var' : {}
#         }
#         self.prob1_sumary_stats = {
#             'sample_number' : 0,
#             'Mean': {},
#             'Var' : {}
#         }
#         self.prob2_sumary_stats = {
#             'sample_number' : 0,
#             'Mean': {},
#             'Var' : {}
#         }
#         self.prob3_sumary_stats = {
#             'sample_number' : 0,
#             'Mean': {},
#             'Var' : {}
#         }

#         self.printer = printer

#     # ===== 主功能方法 =====
#     def update(self, **kwargs):
#         """Observer接口实现, 响应模型训练或验证事件"""
#         if 'valid_preds' in kwargs:
#             mut_labels = self.extract_mut_labels(kwargs['label'])
#             mut_labels = mut_labels.view(-1)
#             preds_each_model = self.extract_sub_model_preds(kwargs['valid_preds'])
#             if preds_each_model:
#                 self.record(preds_each_model, mut_labels)

#         if 'valid_step_finish' in kwargs:
#             self.report_contributions()
#             self.reset()

#     # ===== 核心逻辑 =====
#     def record(self, preds_each_model: dict, mut_labels: torch.Tensor):
#         """记录每轮验证子模型贡献"""
#         num_classes = 4
#         for group in range(num_classes):
#             mut_labels_group_idx = (mut_labels == group)
#             sample_number = np.sum(mut_labels_group_idx)
#             if sample_number == 0:
#                 continue
        
#             prob_sumary_stats = self.choice_prob_sumary_stats(group)
#             if prob_sumary_stats['sample_number'] == 0:
#                 prob_sumary_stats['sample_number'] = sample_number

#             for model_name, preds in preds_each_model.items():
#                 if model_name == 'fused_pred':
#                     continue
#                 preds_one_type = preds[mut_labels_group_idx]
#                 batch_mean, batch_var = self._calc_batch_stats(preds_one_type) 

#                 if model_name not in prob_sumary_stats['Mean']:
#                     prob_sumary_stats['Mean'][model_name] = batch_mean
#                     prob_sumary_stats['Var'][model_name] = batch_var

#                 # 更新均值、方差、样本数量
#                 self._update_contributions(
#                     model_name,
#                     batch_mean=batch_mean,
#                     batch_var=batch_var,
#                     batch_counts=sample_number,
#                     prob_sumary_stats=prob_sumary_stats
#                 )

#             prob_sumary_stats['sample_number'] += sample_number

#     def choice_prob_sumary_stats(self, group):
#         if group == 0:
#             return self.prob0_sumary_stats
#         elif group == 1:
#             return self.prob1_sumary_stats
#         elif group == 2:
#             return self.prob2_sumary_stats
#         elif group == 3:
#             return self.prob3_sumary_stats
#         else:
#             raise ValueError("Error: Unsupported group number for choice_prob_sumary_stats")

#     def report_contributions(self):
#         """输出每个子模型的贡献信息"""
#         for idx, prob_sumary_stats in enumerate([self.prob0_sumary_stats, self.prob1_sumary_stats, self.prob2_sumary_stats, self.prob3_sumary_stats]):
#             if prob_sumary_stats['sample_number'] == 0:
#                 self.printer(f"No contributions recorded in prob{idx}_sumary_stats.")
#                 continue
#             self._print_contribution(f"Mean absolute contribution for prob{idx}", prob_sumary_stats['Mean'])
#             self._print_contribution(f"Variance of contribution for prob{idx}", prob_sumary_stats['Var'])

#     def reset(self):
#         """重置贡献数据"""
#         self.mean_contributions = {}
#         self.var_contributions = {}
#         self.batch_sizes = {}

#     # ===== 私有工具方法 =====
#     def _calc_batch_stats(self, preds):
#         """计算当前批次的均值、方差和样本计数"""
#         preds = self._convert_to_numpy(preds)
#         batch_mean = np.mean(class_preds, axis=0)
#         batch_var = np.var(class_preds, axis=0)
#         return batch_mean, batch_var

#     def _update_specify_model_contributions(self, model_name, batch_mean, batch_var, batch_counts, prob_sumary_stats):
#         """累积更新均值和方差"""
#         existing_mean = prob_sumary_stats['Mean'][model_name]
#         existing_var = prob_sumary_stats['Var'][model_name]
#         existing_counts = prob_sumary_stats['sample_number']

#         total_counts = existing_counts + batch_counts
#         mean_diff = batch_mean - existing_mean

#         # 更新均值
#         prob_sumary_stats['Mean'][model_name] = (
#             existing_mean + (batch_counts / total_counts) * mean_diff
#         )

#         # 更新方差
#         prob_sumary_stats['Var'][model_name] = (
#             (existing_counts * (existing_var + mean_diff**2) +
#              batch_counts * (batch_var + mean_diff**2)) / total_counts
#         )


#     def _print_contribution(self, title, contributions):
#         """格式化输出贡献数据"""
#         self.printer(f"{title}:")
#         for model_name, contribution in contributions.items():
#             self.printer(f"  {model_name}: {contribution}")

#     def _convert_to_numpy(self, tensor):
#         """将张量转换为numpy数组"""
#         if hasattr(tensor, 'is_cuda') and tensor.is_cuda:
#             return tensor.cpu().numpy()
#         return tensor.numpy()

#     # ===== 数据提取方法 =====
#     def extract_mut_labels(self, label):
#         """从标签数据中提取突变标签"""
#         if isinstance(label, dict):
#             return label['label']
#         return label

#     def extract_sub_model_preds(self, preds):
#         """根据数据格式提取子模型预测"""
#         if isinstance(preds[0], dict):
#             return self._extract_from_dict(preds[0])
#         if isinstance(preds, tuple):
#             return self._extract_from_tuple(preds)
#         return None

#     def _extract_from_dict(self, preds):
#         """处理字典格式的子模型预测"""
#         keys = ['local', 'local2', 'local3', 'mid', 'distal', 'out']
#         available_keys = {k: preds[k] for k in keys if preds.get(k) is not None}
#         available_keys['fused_pred'] = available_keys.pop('out', None)
#         return available_keys

#     def _extract_from_tuple(self, preds):
#         """处理元组格式的子模型预测"""
#         if len(preds) == 3:
#             preds_local, preds_distal, pred = preds
#             return {'local': preds_local, 'distal': preds_distal, 'fused_pred': pred}
#         if len(preds) == 4:
#             preds_local, preds_mid, preds_distal, pred = preds
#         elif len(preds) == 5:
#             preds_local, preds_mid, preds_distal, pred, _ = preds
#         else:
#             raise ValueError("Unsupported tuple format for sub-model predictions.")
#         return {'local': preds_local, 'mid': preds_mid, 'distal': preds_distal, 'fused_pred': pred}

class ContributionMinor2(Observer):
    """
    优化代码以支持批次累积计算贡献。
    使用分批均值和方差计算整体方差的方式。
    """

    def __init__(self, printer=print):
        self.prob_summary_stats = [
            {'sample_number': 0, 'Mean': {}, 'Var': {}} for _ in range(4)
        ]
        self.printer = printer

    # ===== 主功能方法 =====
    def update(self, **kwargs):
        """Observer接口实现, 响应模型训练或验证事件"""
        if 'valid_preds' in kwargs:
            mut_labels = self._extract_mut_labels(kwargs['label']).view(-1)
            preds_each_model = self._extract_sub_model_preds(kwargs['valid_preds'])
            if preds_each_model:
                self._record(preds_each_model, mut_labels)

        if 'valid_step_finish' in kwargs:
            self._report_contributions()
            self._reset()

    # ===== 核心逻辑 =====
    def _record(self, preds_each_model: dict, mut_labels: torch.Tensor):
        """记录每轮验证子模型贡献"""
        num_classes = len(self.prob_summary_stats)
        for group in range(num_classes):
            mut_labels_group_idx = (mut_labels == group)
            sample_number = mut_labels_group_idx.sum().item()
            if sample_number == 0:
                continue

            prob_summary = self.prob_summary_stats[group]

            for model_name, preds in preds_each_model.items():
                if model_name == 'fused_pred':
                    continue

                preds_one_type = preds[mut_labels_group_idx]
                batch_mean, batch_var = self._calc_batch_stats(preds_one_type)

                if model_name not in prob_summary['Mean']:
                    prob_summary['Mean'][model_name] = batch_mean
                    prob_summary['Var'][model_name] = batch_var
                else:

                    self._update_contributions(
                        model_name, batch_mean, batch_var, sample_number, prob_summary
                    )
            prob_summary['sample_number'] += sample_number
            

    def _update_contributions(self, model_name, batch_mean, batch_var, batch_counts, prob_summary):
        """累积更新均值和方差"""
        existing_mean = prob_summary['Mean'][model_name]
        existing_var = prob_summary['Var'][model_name]
        existing_counts = prob_summary['sample_number']

        total_counts = existing_counts + batch_counts
        mean_diff = batch_mean - existing_mean

        prob_summary['Mean'][model_name] += (batch_counts / total_counts) * mean_diff
        prob_summary['Var'][model_name] = (
            (existing_counts * (existing_var + mean_diff**2) +
             batch_counts * (batch_var + mean_diff**2)) / total_counts
        )

    def _report_contributions(self):
        """输出每个子模型的贡献信息"""
        for idx, prob_summary in enumerate(self.prob_summary_stats):
            if prob_summary['sample_number'] == 0:
                self.printer(f"No contributions recorded in prob{idx}.")
                continue
            self._print_contribution(f"Mean absolute contribution for prob{idx}", prob_summary['Mean'])
            self._print_contribution(f"Variance of contribution for prob{idx}", prob_summary['Var'])

    def _reset(self):
        """重置贡献数据"""
        for prob_summary in self.prob_summary_stats:
            prob_summary['sample_number'] = 0
            prob_summary['Mean'].clear()
            prob_summary['Var'].clear()

    # ===== 私有工具方法 =====
    def _calc_batch_stats(self, preds):
        """计算当前批次的均值、方差"""
        preds_np = self._convert_to_numpy(preds)
        return preds_np.mean(axis=0), preds_np.var(axis=0)

    def _print_contribution(self, title, contributions):
        """格式化输出贡献数据"""
        #self.printer(f"{title}:")
        for model_name, value in contributions.items():
            value = '\t'.join([str(x) for x in value])
            line = title + f"  ({model_name}): {value}"
            self.printer(line)
            #self.printer(f"  {model_name}: {value}")

    def _convert_to_numpy(self, tensor):
        """将张量转换为numpy数组"""
        if hasattr(tensor, 'is_cuda') and tensor.is_cuda:
            return tensor.cpu().numpy()
        return tensor.numpy()

    # ===== 数据提取方法 =====
    def _extract_mut_labels(self, label):
        """从标签数据中提取突变标签"""
        return label['label'] if isinstance(label, dict) else label

    def _extract_sub_model_preds(self, preds):
        """提取子模型预测"""
        if isinstance(preds[0], dict):
            return {k: v for k, v in preds[0].items() if v is not None}
        if isinstance(preds, tuple):
            return self._extract_from_tuple(preds)
        return None

    def _extract_from_tuple(self, preds):
        """处理元组格式的子模型预测"""
        length_map = {
            3: ('local', 'distal', 'fused_pred'),
            4: ('local', 'mid', 'distal', 'fused_pred'),
            5: ('local', 'mid', 'distal', 'fused_pred', '_')
        }
        if len(preds) in length_map:
            return dict(zip(length_map[len(preds)], preds))
        raise ValueError("Unsupported tuple format for sub-model predictions.")