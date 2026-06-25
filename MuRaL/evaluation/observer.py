import logging
import numpy as np
import torch

logger = logging.getLogger('mural')
from MuRaL.evaluation.gradient_utils import print_gradients, print_gradient_norms

from typing import Dict



class Observer:
    def update(self, **kwargs):
        raise NotImplementedError


class TimeMinor(Observer):
    def __init__(self, out_after_n_batch, dataset_class='Training'):
        self.batch_load_times = []
        self.batch_train_times = []
        self.batch_total_times = []
        self.counter = 0
        self.out_after_n_batch = out_after_n_batch
        self.dataset_class = dataset_class
    
    def record_batch_load(self, time):
        self.batch_load_times.append(time)

    def record_batch_train_times(self, time):
        self.batch_train_times.append(time)

    def record_batch_total_time(self, time):
        self.batch_total_times.append(time)

    def out_batch_times(self):
        logger.debug("%s load %d batch used %.2f min", self.dataset_class, self.out_after_n_batch, np.sum(self.batch_load_times) / 60)
        logger.debug("%s train %d batch used %.2f min", self.dataset_class, self.out_after_n_batch, np.sum(self.batch_train_times) / 60)
        logger.debug("%s after %d batch used %.2f min", self.dataset_class, self.out_after_n_batch, np.sum(self.batch_total_times) / 60)
        self.reset()

    def reset(self):
        self.counter = 0
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
            self.counter += 1
            if self.counter == self.out_after_n_batch:
                self.out_batch_times()

class GradMinor(Observer):
    def __init__(self, out_after_n_batch, first_epoch=True):
        self.counter = 0
        self.out_after_n_batch = out_after_n_batch
        if first_epoch:
            self.out_epoch = 5
        else:
            self.out_epoch = 0
    
    def out_grad(self, model):
        logger.debug("Layer-wise Gradient Distribution:")
        print_gradients(model, print=logger.debug)
        print_gradient_norms(model, print=logger.debug)
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
        
class UniversalLossRecorder:
    """Records and reports loss breakdowns from any strategy.

    Works with both modern dict-style losses and legacy tuple-style losses.
    Keys are discovered dynamically from the first recorded batch.
    """

    def __init__(self):
        self.keys = None
        self._init_buffers()

    def _init_buffers(self):
        # dict-mode: dynamic accumulators keyed by loss dict keys
        self._buffers = {}
        # tuple-mode: positional accumulators for legacy strategies
        self._tuple_losses = []

    def record(self, loss):
        if isinstance(loss, dict):
            self._record_dict(loss)
        elif isinstance(loss, torch.Tensor):
            self._tuple_losses.append(loss.item())
        elif isinstance(loss, (tuple, list)):
            self._tuple_losses.append([v.item() if isinstance(v, torch.Tensor) else v for v in loss])

    def _record_dict(self, loss):
        if self.keys is None:
            self.keys = list(loss.keys())
            for k in self.keys:
                self._buffers[k] = []
        for k in self.keys:
            v = loss.get(k)
            if v is not None:
                self._buffers[k].append(v.item())

    def reset(self):
        self.keys = None
        self._init_buffers()

    def out_mean_loss(self, dataset_class, sample_number):
        if self._tuple_losses:
            total = self._out_tuple_loss(dataset_class, sample_number)
        else:
            total = self._out_dict_loss(dataset_class, sample_number)
        return {'loss': total / sample_number if sample_number else 0}

    def _out_dict_loss(self, dataset_class, sample_number):
        total = 0
        for k in self.keys:
            if self._buffers[k]:
                arr = np.array(self._buffers[k])
                logger.info("%s %s: %.4f; Batch Var: %.4f", dataset_class, k, arr.sum() / sample_number, arr.var())
        k = 'total_loss' if 'total_loss' in (self.keys or []) else 'loss'
        if self.keys and k in self.keys and self._buffers.get(k):
            total = np.sum(self._buffers[k])
        elif self.keys:
            for k in self.keys:
                if self._buffers.get(k):
                    total = np.sum(self._buffers[k])
                    break
        return total

    def _out_tuple_loss(self, dataset_class, sample_number):
        first = self._tuple_losses[0]
        if isinstance(first, list):
            labels = {3: ['Local', 'Distal', 'Total'],
                      4: ['Local', 'Mid', 'Distal', 'Total'],
                      5: ['Local', 'Mid', 'Distal', 'Total', 'Construct']}
            names = labels.get(len(first), [f'loss_{i}' for i in range(len(first))])
            for i, name in enumerate(names):
                vals = [row[i] for row in self._tuple_losses]
                logger.info("%s %s Loss: %.4f; Batch Var: %.4f", dataset_class, name, np.sum(vals) / sample_number, np.var(vals))
            return np.sum([row[-1] for row in self._tuple_losses])
        else:
            total = np.sum(self._tuple_losses)
            logger.info("%s Total Loss: %.4f; Batch Var: %.4f", dataset_class, total / sample_number, np.var(self._tuple_losses))
            return total


class LossMinor(Observer):
    def __init__(self):
        self.loss_strategy = UniversalLossRecorder()
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
            self.out_mean_losses(dataset_class="Training", sample_number=self.sample_number)
            return {}
        if 'valid_step_finish' in kwargs:
            minor_dict = self.out_mean_losses(dataset_class="Validation", sample_number=self.sample_number)
            return {'valid_loss': minor_dict['loss']}

        

class ModelSaverObserve(Observer):
    def __init__(self, model_saver):
        self.model_saver = model_saver

    def update(self, **kwargs):
        if 'epoch_finish' in kwargs:
            self.model_saver.save_model(kwargs['epoch_finish'])
            logger.info("Model saved at epoch %s", kwargs['epoch_finish'])

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
        if isinstance(preds, tuple):
            return preds[0]['out']
        return preds['out']

    
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
            return
        self.mu = mu if self.mu is None else torch.cat([self.mu, mu], dim=0)
        self.r = r if self.r is None else torch.cat([self.r, r], dim=0)

    def output(self):
        mu, r = self.mu, self.r
        self.reset()
        return mu, r

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            self.recode(kwargs['valid_preds'])


class DirMDNRecoder(Observer):
    """Collect evidence and unactivated components (pi_logits, alpha_raw)
    during DirMDN validation/prediction.

    - output() returns evidence
    - get_components() returns (pi_logits, alpha_raw)
    - reset() clears all buffers (caller's responsibility)
    - output() and get_components() do NOT reset -- callable in any order
    """

    def __init__(self):
        super().__init__()
        self.evidence = None
        self.pi_logits = None
        self.alpha_raw = None

    def reset(self):
        self.evidence = None
        self.pi_logits = None
        self.alpha_raw = None

    def recode(self, preds):
        predict_out, _ = preds if isinstance(preds, tuple) else (preds, None)
        if 'pi_logits' not in predict_out:
            return
        from MuRaL.models.dirichlet_mdn_model import dirichlet_mdn_predict_from_output
        with torch.no_grad():
            result = dirichlet_mdn_predict_from_output(predict_out)
            ev = result['evidence'].detach().cpu()
        self.evidence = (
            ev if self.evidence is None
            else torch.cat([self.evidence, ev], dim=0)
        )
        pi = predict_out['pi_logits'].detach().cpu()
        alpha = predict_out['alpha_raw'].detach().cpu()
        self.pi_logits = (
            pi if self.pi_logits is None
            else torch.cat([self.pi_logits, pi], dim=0)
        )
        self.alpha_raw = (
            alpha if self.alpha_raw is None
            else torch.cat([self.alpha_raw, alpha], dim=0)
        )

    def output(self):
        return self.evidence

    def get_components(self):
        return self.pi_logits, self.alpha_raw

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            self.recode(kwargs['valid_preds'])


class GammaMDNRecoder(Observer):
    """Collect pi_entropy and unactivated components (pi_logits, alpha_raw, beta_raw)
    during Gamma MDN validation/prediction.

    - output() returns pi_entropy
    - get_components() returns (pi_logits, alpha_raw, beta_raw)
    - reset() clears all buffers (caller's responsibility)
    - output() and get_components() do NOT reset -- callable in any order
    """

    def __init__(self):
        super().__init__()
        self.pi_entropy = None
        self.pi_logits = None
        self.alpha_raw = None
        self.beta_raw = None

    def reset(self):
        self.pi_entropy = None
        self.pi_logits = None
        self.alpha_raw = None
        self.beta_raw = None

    def recode(self, preds):
        predict_out, _ = preds if isinstance(preds, tuple) else (preds, None)
        if 'pi_logits' not in predict_out:
            return
        from MuRaL.models.gamma_mdn_model import compute_mdn_uncertainty
        with torch.no_grad():
            uncertainty = compute_mdn_uncertainty(predict_out)
            entropy = uncertainty['pi_entropy'].cpu()
        self.pi_entropy = (
            entropy if self.pi_entropy is None
            else torch.cat([self.pi_entropy, entropy], dim=0)
        )
        pi = predict_out['pi_logits'].detach().cpu()
        alpha = predict_out['alpha_raw'].detach().cpu()
        beta = predict_out['beta_raw'].detach().cpu()
        self.pi_logits = (
            pi if self.pi_logits is None
            else torch.cat([self.pi_logits, pi], dim=0)
        )
        self.alpha_raw = (
            alpha if self.alpha_raw is None
            else torch.cat([self.alpha_raw, alpha], dim=0)
        )
        self.beta_raw = (
            beta if self.beta_raw is None
            else torch.cat([self.beta_raw, beta], dim=0)
        )

    def output(self):
        return self.pi_entropy

    def get_components(self):
        return self.pi_logits, self.alpha_raw, self.beta_raw

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

    def _extract_sub_model_preds(self, preds):
        return _extract_model_components(preds)




# ---- shared helper --------------------------------------------------

def _extract_model_components(preds):
    """Normalise model output to {component_name: Tensor} dict.

    Works with both the normalised (PredictOutput, Optional[SegmentOutput])
    format and legacy dict/tuple formats.
    """
    # normalised format: (PredictOutput, Optional[SegmentOutput])
    if isinstance(preds, tuple) and hasattr(preds[0], 'items'):
        return {k: v for k, v in preds[0].items() if v is not None}
    # legacy dict-in-tuple / PredictOutput-in-tuple
    if isinstance(preds, tuple) and len(preds) >= 1 and hasattr(preds[0], 'items'):
        return {k: v for k, v in preds[0].items() if v is not None}
    # legacy positional tuple: (local, distal, fused) or (local, mid, distal, fused)
    if isinstance(preds, tuple) and len(preds) >= 3:
        names = {3: ('local', 'distal', 'fused_pred'),
                 4: ('local', 'mid', 'distal', 'fused_pred'),
                 5: ('local', 'mid', 'distal', 'fused_pred', '_')}
        return dict(zip(names.get(len(preds), ()), preds))
    return None


class ContributionMinor2(Observer):
    """
    优化代码以支持批次累积计算贡献。
    使用分批均值和方差计算整体方差的方式。
    """

    def __init__(self):
        self.prob_summary_stats = [
            {'sample_number': 0, 'Mean': {}, 'Var': {}} for _ in range(4)
        ]
        

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
                logger.debug("No contributions recorded in prob%d.", idx)
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
            logger.debug(line)
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
        return _extract_model_components(preds)


class GammaTotalDirichletRecoder(Observer):
    """Collect pi_entropy and raw parameters during Gamma-Total-Dirichlet MDN prediction.

    - output() returns pi_entropy
    - get_components() returns (gamma_alpha_raw, gamma_beta_raw, dir_alpha_raw)
    - reset() clears all buffers (caller's responsibility)
    """

    def __init__(self):
        super().__init__()
        self.pi_entropy = None
        self.pi_logits = None
        self.gamma_alpha_raw = None
        self.gamma_beta_raw = None
        self.dir_alpha_raw = None

    def reset(self):
        self.pi_entropy = None
        self.pi_logits = None
        self.gamma_alpha_raw = None
        self.gamma_beta_raw = None
        self.dir_alpha_raw = None

    def recode(self, preds):
        predict_out, _ = preds if isinstance(preds, tuple) else (preds, None)
        if 'pi_logits' not in predict_out:
            return
        from MuRaL.models.gamma_mdn_model import compute_mdn_uncertainty
        with torch.no_grad():
            uncertainty = compute_mdn_uncertainty(predict_out)
            entropy = uncertainty['pi_entropy'].cpu()
        self.pi_entropy = (
            entropy if self.pi_entropy is None
            else torch.cat([self.pi_entropy, entropy], dim=0)
        )
        pi = predict_out['pi_logits'].detach().cpu()
        self.pi_logits = (
            pi if self.pi_logits is None
            else torch.cat([self.pi_logits, pi], dim=0)
        )
        ga = predict_out['gamma_alpha_raw'].detach().cpu()
        gb = predict_out['gamma_beta_raw'].detach().cpu()
        da = predict_out['dir_alpha_raw'].detach().cpu()
        self.gamma_alpha_raw = (
            ga if self.gamma_alpha_raw is None
            else torch.cat([self.gamma_alpha_raw, ga], dim=0)
        )
        self.gamma_beta_raw = (
            gb if self.gamma_beta_raw is None
            else torch.cat([self.gamma_beta_raw, gb], dim=0)
        )
        self.dir_alpha_raw = (
            da if self.dir_alpha_raw is None
            else torch.cat([self.dir_alpha_raw, da], dim=0)
        )

    def output(self):
        return self.pi_entropy

    def get_components(self):
        return self.pi_logits, self.gamma_alpha_raw, self.gamma_beta_raw, self.dir_alpha_raw

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            self.recode(kwargs['valid_preds'])


class GammaLambdaRecoder(Observer):
    """Collect mixture-weighted mutation intensity lambda during validation.

    λ_{k,i} = α_{k,i} / β_{k,i}, then pi-weighted sum to (B, C).
    Supports both softplus and log-parameterized activation.
    """

    def __init__(self, gamma_activation='softplus'):
        super().__init__()
        self.lam = None
        self.gamma_activation = gamma_activation

    def reset(self):
        self.lam = None

    def recode(self, preds):
        import torch.nn.functional as F
        predict_out, _ = preds if isinstance(preds, tuple) else (preds, None)
        if 'alpha_raw' not in predict_out:
            return

        alpha_raw = predict_out['alpha_raw'].detach()
        beta_raw = predict_out['beta_raw'].detach()

        if self.gamma_activation == 'log':
            from MuRaL.models.gamma_mdn_model import activate_gamma_alpha_beta
            alpha, beta = activate_gamma_alpha_beta(alpha_raw, beta_raw)
        else:
            alpha = F.softplus(alpha_raw) + 1e-8
            beta = F.softplus(beta_raw) + 1e-8

        lam = alpha / beta                                              # (B, K, C)
        pi = F.softmax(predict_out['pi_logits'].detach(), dim=1)       # (B, K)
        lam_mix = (pi.unsqueeze(-1) * lam).sum(dim=1).cpu()            # (B, C)

        self.lam = lam_mix if self.lam is None else torch.cat([self.lam, lam_mix], dim=0)

    def output(self):
        return self.lam

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            self.recode(kwargs['valid_preds'])


class GammaLambdaAlphaRecoder(Observer):
    """Collect pi_entropy and raw parameters for (λ,α)-parameterized models.

    - output() returns pi_entropy
    - get_components() returns (pi_logits, lambda_raw, alpha_raw, dir_alpha_raw)
    """

    def __init__(self):
        super().__init__()
        self.pi_entropy = None
        self.pi_logits = None
        self.lambda_raw = None
        self.alpha_raw = None
        self.dir_alpha_raw = None

    def reset(self):
        self.pi_entropy = None
        self.pi_logits = None
        self.lambda_raw = None
        self.alpha_raw = None
        self.dir_alpha_raw = None

    def recode(self, preds):
        predict_out, _ = preds if isinstance(preds, tuple) else (preds, None)
        if 'pi_logits' not in predict_out:
            return
        from MuRaL.models.gamma_mdn_model import compute_mdn_uncertainty
        with torch.no_grad():
            uncertainty = compute_mdn_uncertainty(predict_out)
            entropy = uncertainty['pi_entropy'].cpu()
        self.pi_entropy = (
            entropy if self.pi_entropy is None
            else torch.cat([self.pi_entropy, entropy], dim=0)
        )
        pi = predict_out['pi_logits'].detach().cpu()
        self.pi_logits = (
            pi if self.pi_logits is None
            else torch.cat([self.pi_logits, pi], dim=0)
        )
        lr = predict_out['lambda_raw'].detach().cpu()
        ar = predict_out['alpha_raw'].detach().cpu()
        da = predict_out['dir_alpha_raw'].detach().cpu()
        self.lambda_raw = (
            lr if self.lambda_raw is None
            else torch.cat([self.lambda_raw, lr], dim=0)
        )
        self.alpha_raw = (
            ar if self.alpha_raw is None
            else torch.cat([self.alpha_raw, ar], dim=0)
        )
        self.dir_alpha_raw = (
            da if self.dir_alpha_raw is None
            else torch.cat([self.dir_alpha_raw, da], dim=0)
        )

    def output(self):
        return self.pi_entropy

    def get_components(self):
        return self.pi_logits, self.lambda_raw, self.alpha_raw, self.dir_alpha_raw

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            self.recode(kwargs['valid_preds'])