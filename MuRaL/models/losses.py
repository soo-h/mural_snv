import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import inspect


class NegativeBinomialLoss(nn.Module):
    """负二项分布负对数似然损失。

    μ 和 r 必须由模型通过 forward 输出已激活的正值。
    Loss 自身不维护可学习参数，仅计算 NLL 公式。
    """

    def __init__(self, n_class, reduction='sum'):
        super().__init__()
        self.n_class = n_class
        self.reduction = reduction

    def forward(self, mu, target, r):
        """
        Args:
            mu: 模型已激活的 μ（正值，[batch, n_class]）
            target: 类别索引 [batch] 或已处理计数 [batch, n_class]
            r: 模型已激活的 r（正值，[batch, n_class]）
        """
        if target.dim() == 1:
            target = F.one_hot(target, self.n_class).float()

        # NLL = -log P
        nll = (
            -torch.lgamma(target + r)
            + torch.lgamma(r)
            + torch.lgamma(target + 1)
            + target * torch.log1p(r / (mu + 1e-8))
            + r * torch.log1p(mu / (r + 1e-8))
        )

        if self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'mean':
            return nll.mean()
        return nll


class LossFactory():
    def __init__(self) -> None:
        pass

    def create_loss(self, loss_name=None, use_sample_weight=False, n_class=None):
        if loss_name is None or loss_name == 'CrossEntropy':
            if use_sample_weight:
                return torch.nn.CrossEntropyLoss(reduction='none')
            else:
                return torch.nn.CrossEntropyLoss(reduction='sum')
        elif loss_name == 'NegBinomial':
            if n_class is None:
                raise ValueError("n_class is required for NegBinomial loss")
            # NB loss 始终用 reduction='sum'，sample_weight 语义由调用方处理
            return NegativeBinomialLoss(n_class=n_class, reduction='sum')
        else:
            raise ValueError(f"Unknown loss_name: {loss_name}")

def compute_local_distal_loss(preds, y, criterion):
    preds_local, preds_distal, preds = preds
    loss_local = criterion(preds_local, y)
    loss_distal = criterion(preds_distal, y)
    loss = criterion(preds, y)
    return loss_local, loss_distal, loss

def compute_complex_loss_with_decoder(preds, y, criterion):
    preds_local, preds_mid, preds_distal, preds, loss_construct = preds
    loss_local = criterion(preds_local, y)
    loss_mid = criterion(preds_mid, y)
    loss_distal = criterion(preds_distal, y)
    if isinstance(loss_construct, tuple):
        loss_construct, loss_descrim = loss_construct

    loss = criterion(preds, y)
    return loss_local, loss_mid, loss_distal, loss, loss_construct

def compute_mid_distal_loss(preds, y, criterion):
    """处理中间和远程预测的损失计算"""
    preds_local, preds_mid, preds_distal, preds = preds
    loss_local = criterion(preds_local, y)
    loss_mid = criterion(preds_mid, y)
    loss_distal = criterion(preds_distal, y)
    loss = criterion(preds, y)
    return loss_local, loss_mid, loss_distal, loss    

class LossCalcStrategyFactory:
    @staticmethod
    def get_loss_strategy(strategy_name=None, **kwargs):
        loss_calc_strategy_name = {
            'OnlyCombined': OnlyCombinedLossStrategy,
            'LocalDistalCombined': LocalDistalCombinedLossStrategy,
            'LocalMidDistalCombined': LocalMidDistalCombinedLossStrategy,
            'LocalMidDistalCombinedDecoder': LocalMidDistalCombinedDecoderLossStrategy,
            'segment_soft_label' : SegmentCombinedLossStrategy,
            'segment_soft_label_step' : SegmentCombinedLossStrategy,
            'segment_soft_label_step_withGAN' : SegmentCombinedWithGANLossStrategy,

            'AvgSegmentLabel_withGAN' : SoftLabelUtilAvgSegmentWithGANLossStrategy,
            'AvgSegmentLabel_withGAN2' : SoftLabelUtilAvgSegmentWithGANLossStrategy2,
            'AvgSegMutUseInLocal' : AdaptiveLossStrategy2,
            'AvgSegMutAndKmerMut' : SegmentCombinedLossStrategy,
            
            'AvgSegMutAndNucSkewUseInLocal' : AdaptiveLossStrategy2,
            'AvgSegMutAndKmerMutUseInLocal' : AdaptiveLossStrategy2,
            'AvgStepMutAndKmerMutUseInLocal' : AdaptiveLossStrategy2,
            'SKA_local' : AdaptiveLossStrategy2,

            'AvgStepMutAndKmerMutCominedLoss': CombinedAvgMutLoss,
        }
        # # adapt Original test
        # if strategy_name is None:
        #     return AdaptiveLossStrategy()
        # else:
        #     return loss_calc_startegy_name[strategy_name]()
        # adapt Original test
        if strategy_name is None:
            selected_class = AdaptiveLossStrategy2
        elif strategy_name in loss_calc_strategy_name:
            selected_class = loss_calc_strategy_name[strategy_name]
        else:
            raise ValueError(f"Unknown strategy_name: {strategy_name}")

        # 获取类的构造函数参数
        signature = inspect.signature(selected_class.__init__)
        class_params = signature.parameters

        # 确定必需参数（没有默认值的参数）
        required_params = [
            name for name, param in class_params.items()
            if param.default == inspect.Parameter.empty and name != 'self'
        ]

        # 过滤出 kwargs 中匹配的参数
        filtered_kwargs = {
            key: value for key, value in kwargs.items()
            if key in class_params and key != 'self'  # 排除 `self` 参数
        }

        # 检查是否提供了所有必需参数
        missing_params = [param for param in required_params if param not in filtered_kwargs]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {selected_class.__name__}: {', '.join(missing_params)}"
            )

        # 实例化类并传递过滤后的参数
        return selected_class(**filtered_kwargs)


class AdaptiveLossStrategy():
    def __init__(self) -> None:
        self.loss = None 
    
    def calc_loss(self, preds, y, criterion):
        if isinstance(preds, tuple):
            if len(preds) == 3:
                res = compute_local_distal_loss(preds, y, criterion) 
                self.loss = res[-1]
                return res

            elif len(preds) >= 4:
                if len(preds) == 5:
                    res = compute_complex_loss_with_decoder(preds, y, criterion)
                    self.loss = res[-2] + res[-1]
                    return res
                else:
                    res = compute_mid_distal_loss(preds, y, criterion)
                    self.loss = res[-1]
                    return res
            else:
                sys.exit("Error: unconsider situation, check AdaptiveLossStrategy")
        else:
            res = criterion(preds, y)
            return res
    
    def extract_total_loss(self):
        return self.loss
    
    def extract_pred(self, preds):
        if isinstance(preds, tuple):
            if len(preds) == 3:
                preds_local, preds_distal, preds = preds
            if len(preds) == 4:
                preds_local, preds_mid, preds_distal, preds = preds
            if len(preds) == 5:
                preds_local, preds_mid, preds_distal, preds, loss_construct = preds

        return preds

class AdaptiveLossStrategy2():
    def __init__(self) -> None:
        self.total_loss = None

    def calc_loss(self, preds, labels, criterion, sample_weight=None):

        preds, segment_preds = preds
        self.check_preds(preds)
        self.check_labels(labels)

        y = labels.get('label')

        is_nb_loss = isinstance(criterion, NegativeBinomialLoss)
        if is_nb_loss:
            mu = preds.get('mu')
            r = preds.get('r')
            if mu is None or r is None:
                raise ValueError(
                    "NegativeBinomialLoss requires model to output 'mu' and 'r'. "
                    "Use a model with r_head (127_nb, 127_nb_v2, 127_nb_v3, etc.)."
                )

        # 辅助函数：计算加权或普通损失
        def _calc_loss(pred_or_mu, target, r=None):
            if pred_or_mu is None:
                return None
            if is_nb_loss:
                # sample_weight 语义：替换 one-hot 中的计数
                if sample_weight is not None:
                    target = F.one_hot(target, criterion.n_class).float()
                    target = target * sample_weight.unsqueeze(-1)
                return criterion(pred_or_mu, target, r)
            loss = criterion(pred_or_mu, target)
            if sample_weight is not None and loss.dim() > 0:
                return (loss * sample_weight.squeeze()).sum()
            return loss

        if is_nb_loss:
            # NB loss：只有 out 有已激活的 mu/r，其余子模型输出 raw logits → None
            loss = _calc_loss(mu, y, r=r)
            loss_local1 = _calc_loss(None, y)
            loss_local2 = _calc_loss(None, y)
            loss_local3 = _calc_loss(None, y)
            loss_mid = _calc_loss(None, y)
            loss_distal = _calc_loss(None, y)
            loss_arg_feature = _calc_loss(None, y)
            loss_dual_head = 0
        else:
            # CE loss：分量正常计算
            mid = preds.get('mid')
            distal = preds.get('distal')
            out = preds.get('out')

            local1 = preds.get('local')
            loss_local1 = _calc_loss(local1, y)

            local2 = preds.get('local2')
            loss_local2 = _calc_loss(local2, y)

            local3 = preds.get('local3')
            loss_local3 = _calc_loss(local3, y)

            loss_mid = _calc_loss(mid, y)

            loss_distal = _calc_loss(distal, y)

            arg_feature = preds.get('arg_feature')
            loss_arg_feature = _calc_loss(arg_feature, y)

            loss_dual_head = 0
            if 'local_h1' in preds:
                assert 'local_h2' in preds, "Both local_h1 and local_h2 should be present"
                loss_local_h1 = _calc_loss(preds['local_h1'], self._to_h1_label(y))
                loss_local_h2 = _calc_loss(preds['local_h2'], self._to_h2_label(y))
                loss_dual_head = loss_local_h1 + loss_local_h2

            loss = _calc_loss(out, y)

        self.total_loss = loss + 0.25 * loss_dual_head

        return {'local_loss': loss_local1,
                'local2_loss' : loss_local2,
                'local3_loss' : loss_local3,
                'mid_loss' : loss_mid,
                'distal_loss' : loss_distal,
                'dual_head_loss' : loss_dual_head,
                'arg_feature_loss' : loss_arg_feature,
                'loss' : self.total_loss} 
    
    def check_preds(self, preds):
        if not isinstance(preds, dict):
            sys.exit(f"Error: pred should be dict type, but input is {type(preds)}")

    def _to_h1_label(self, y):
        """
        7-class → Head1
        {0,4,5,6} → 0
        """
     # 显式映射表，消除所有条件分支
        mapping = torch.tensor(
            [0, 1, 2, 3, 0, 0, 0],
            device=y.device,
            dtype=y.dtype
            )
        return mapping[y]

    def _to_h2_label(self, y):
        """
        7-class → Head2
        {0,1,2,3} → 0
        4 → 1
        5 → 2
        6 → 3
        """
     # 显式映射表，消除所有条件分支
        mapping = torch.tensor(
            [0, 0, 0, 0, 1, 2, 3],
            device=y.device,
            dtype=y.dtype
            )
        return mapping[y]


    def check_labels(self, labels):
        if not isinstance(labels, dict):
            sys.exit(f"Error: labels should be dict type, but input is {type(labels)}")
    def extract_total_loss(self):
        return self.total_loss

class CombinedAvgMutLoss():
    def __init__(self, avg_mut_loss_strategy=None) -> None:
        self.total_loss = None
        self.avg_mut_criterion = nn.MSELoss()
        self.avg_mut_loss_strategy = avg_mut_loss_strategy

    def calc_loss(self, preds, labels, criterion):

        preds, segment_preds = preds
        self.check_preds(preds)
        self.check_labels(labels)

        y = labels.get('label')
        batch_avg_mut = labels.get('avg_mut')
        batch_avg_mut_all_prob = self.calc_obs_all_prob_mut_rate(batch_avg_mut, self.avg_mut_loss_strategy)
        pred_avg_mut_all_prob = self.calc_pred_all_predict_mut_rate(preds, self.avg_mut_loss_strategy)

        main_loss = self.calc_main_loss(preds, y, criterion)
        avg_mut_loss = self.calc_avg_mut_loss(pred_avg_mut_all_prob, batch_avg_mut_all_prob, self.avg_mut_criterion)

        self.total_loss = main_loss.get('loss') + avg_mut_loss.get('avg_mut_loss_total')
        main_loss.update(avg_mut_loss)

        return main_loss

    def calc_main_loss(self, preds, y, criterion):

        local1 = preds.get('local')
        local2 = preds.get('local2')
        local3 = preds.get('local3')
        mid = preds.get('mid')
        distal = preds.get('distal')
        out = preds.get('out')

        loss_local1 = criterion(local1, y) if local1 is not None else None
        loss_local2 = criterion(local2, y) if local2 is not None else None
        loss_local3 = criterion(local3, y) if local3 is not None else None
        loss_mid = criterion(mid, y) if mid is not None else None
        loss_distal = criterion(distal, y) if distal is not None else None
        loss = criterion(out, y)
        return {
            'local1_loss': loss_local1,
            'local2_loss': loss_local2,
            'local3_loss': loss_local3,
            'mid_loss': loss_mid,
            'distal_loss': loss_distal,
            'loss': loss,
        }

    def calc_avg_mut_loss(self, batch_avg_mut_all_prob, batch_avg_mut, criterion):
        loss = criterion(batch_avg_mut_all_prob, batch_avg_mut)
        return {'avg_mut_loss_total' : loss}

    def calc_obs_all_prob_mut_rate(self, batch_avg_mut_all_prob, method):
        # don't consider label
        batch_avg_mut_all_prob = batch_avg_mut_all_prob.sum(dim=1) #(n, 1)
        if method is not None:
            return self._avg_by_index(batch_avg_mut_all_prob, 10)

        return batch_avg_mut_all_prob

    def calc_pred_all_predict_mut_rate(self, pred_avg_mut, method):
        out = pred_avg_mut.get('out')
        if out is None:
            raise ValueError("Error: out is None")
        pred_avg_mut_all_prob = out.sum(dim=1)

        if method is not None:
            return self._avg_by_index(pred_avg_mut_all_prob, 10)
        return pred_avg_mut_all_prob

    def _avg_by_index(self, tensor, n):
        num_full_groups = tensor.size(0) // n
        grouped_tensor = tensor[:num_full_groups * n].view(-1, n).mean(dim=1)  # 完整组的平均
        remainder = tensor[num_full_groups * n:]  # 不完整的部分
        if remainder.numel() > 0:  # 如果有剩余部分
            remainder_mean = remainder.mean().unsqueeze(0)  # 计算剩余部分的平均值
            return torch.cat([grouped_tensor, remainder_mean])  # 拼接完整组和剩余部分
        return grouped_tensor
    
    def check_preds(self, preds):
        if not isinstance(preds, dict):
            sys.exit(f"Error: pred should be dict type, but input is {type(preds)}")

    def check_labels(self, labels):
        if not isinstance(labels, dict):
            sys.exit(f"Error: labels should be dict type, but input is {type(labels)}")
    def extract_total_loss(self):
        return self.total_loss


class OnlyCombinedLossStrategy():
    def __init__(self) -> None:
        self.loss = None

    def calc_loss(self, preds, y, criterion):
        loss = criterion(preds, y)
        self.loss = loss
        return loss
    
    def extract_total_loss(self):
        return self.loss

class LocalDistalCombinedLossStrategy():
    def __init__(self) -> None:
        self.loss = None

    def calc_loss(self, preds, y, criterion):
        loss_local, loss_distal, loss = compute_local_distal_loss(preds, y, criterion)
        self.loss = loss
        return loss_local, loss_distal, loss
    
    def extract_total_loss(self):
        return self.loss

class LocalMidDistalCombinedLossStrategy():
    def __init__(self) -> None:
        self.loss = None

    def calc_loss(self, preds, y, criterion):
        loss_local, loss_mid, loss_distal, loss = compute_mid_distal_loss(preds, y, criterion)
        self.loss = loss
        return loss_local, loss_mid, loss_distal, loss

    def extract_total_loss(self):
        return self.loss

class LocalMidDistalCombinedDecoderLossStrategy():
    def __init__(self) -> None:
        self.loss = None

    def calc_loss(self, preds, y, criterion):
        loss_local, loss_mid, loss_distal, loss, loss_construct = compute_complex_loss_with_decoder(preds, y, criterion)
        self.loss = loss
        return loss_local, loss_mid, loss_distal, loss, loss_construct
    def extract_total_loss(self):
        return self.loss



class SegmentCombinedLossStrategy():
    def __init__(self) -> None:
        self.total_loss = None
        #self.lambda1 = 0.5
        #self.lambda2 = 0.5
        self.lambda_id = 0
        #self.lambda2 = 20
        self.lambda_avg_mut = 2
        self.lambda_avg_kmer_mut = 20

        self.avg_mut_criterion = nn.KLDivLoss(reduction='sum')
        self.avg_kmer_mut_criterion = nn.MSELoss()
        print(f"lambda_avgmut: {self.lambda_avg_mut}; lambda_avg_kmer_mut: {self.lambda_avg_kmer_mut} ; lambda_id: {self.lambda_id}")

    def calc_avgmut_loss(self, predict, y, criterion):
        log_probs = nn.functional.log_softmax(predict, dim=1)
        loss = criterion(log_probs, y)
        return loss   

    def calc_loss(self, preds, labels, criterion):

        self.check_preds(preds)
        self.check_labels(labels)
        self.total_loss = 0

        predict_out, segment_pred = preds

        avg_mut = labels.get('avg_mut')
        pred_avg_mut = segment_pred.get('avg_mut')
        avg_mut_loss = self.calc_avgmut_loss(pred_avg_mut, avg_mut, self.avg_mut_criterion)
        if 'segment_id' in segment_pred:
            segment_id = labels.get('segment_id')
            pred_segment_id = segment_pred.get('segment_id')
            segment_id_loss = self.calc_segment_id_loss(pred_segment_id, segment_id)
            self.total_loss += self.lambda_id * segment_id_loss
        if 'avg_kmer_mut' in segment_pred:
            pred_avg_kmer_mut = segment_pred.get('avg_kmer_mut')
            avg_kmer_mut = labels.get('avg_kmer_mut')
            avg_kmer_mut_loss = self.calc_avgmut_loss(pred_avg_mut, avg_mut, self.avg_mut_criterion)
            self.total_loss += self.lambda_avg_kmer_mut * avg_kmer_mut_loss

        y = labels.get('label')
        loss_local, loss_mid, loss_distal, loss = self.calc_main_loss(predict_out, y, criterion)
        #avg_mut_loss = self.calc_avg_mut_loss(pred_avg_mut, avg_mut, self.criterion)

        self.total_loss += loss 
        self.total_loss += self.lambda2 * avg_mut_loss

        loss_dict = {'local_loss': loss_local, 
                'mid_loss' : loss_mid, 
                'distal_loss' : loss_distal, 
                'loss' : loss, 
                'total_loss' : self.total_loss,}

        if 'avg_kmer_mut' in segment_pred:
            loss_dict['avg_kmer_mut_loss'] = avg_kmer_mut_loss
        if 'segment_id' in segment_pred:
            loss_dict['segment_id_loss'] = segment_id_loss
        return loss_dict


    def calc_main_loss(self, predict_out, y, criterion):
        pred_local = predict_out.get('local')
        pred_mid = predict_out.get('mid')
        pred_distal = predict_out.get('distal')
        pred_total = predict_out.get('out')

        loss_local = criterion(pred_local, y)
        loss_mid = criterion(pred_mid, y)
        loss_distal = criterion(pred_distal, y)
        loss = criterion(pred_total, y)
        
        return loss_local, loss_mid, loss_distal, loss
    
    def calc_segment_id_loss(self, pred_segment_id, segment_id):
        criterion = torch.nn.MSELoss(reduction='sum')
        segment_id_loss = criterion(pred_segment_id, segment_id)
        return segment_id_loss

    def calc_avg_mut_loss(self, pred_avg_mut, avg_mut, criterion):
        avg_mut_loss = criterion(pred_avg_mut, avg_mut)
        return avg_mut_loss
    
    def check_preds(self, preds):
        for pred in preds:
            if not isinstance(pred, dict):
                sys.exit(f"Error: pred should be dict type, but input is {type(pred)}")

    def check_labels(self, labels):
        if not isinstance(labels, dict):
            sys.exit(f"Error: labels should be dict type, but input is {type(labels)}")
    def extract_total_loss(self):
        return self.total_loss

class SegmentCombinedWithGANLossStrategy():
    def __init__(self) -> None:
        self.total_loss = None
        #self.lambda1 = 0.5
        #self.lambda2 = 0.5
        self.lambda1 = 0.01
        self.lambda1 = 0
        self.lambda2 = 2
        self.lambda3 = 1
        print(f"lambda1: {self.lambda1}, lambda2: {self.lambda2}, lambda3: {self.lambda3}")
    
    def calc_loss(self, preds, labels, criterion):

        self.check_preds(preds)
        self.check_labels(labels)

        predict_out, segment_pred, construct_loss = preds

        discrim_loss = construct_loss.get('discrim_loss')
        construct_loss = construct_loss.get('construct_loss')

        pred_avg_mut = segment_pred.get('avg_mut')
        pred_segment_id = segment_pred.get('segment_id')

        y = labels.get('label')
        segment_id = labels.get('segment_id')
        avg_mut = labels.get('avg_mut')

        loss_local, loss_mid, loss_distal, loss = self.calc_main_loss(predict_out, y, criterion)
        segment_id_loss = self.calc_segment_id_loss(pred_segment_id, segment_id)
        avg_mut_loss = self.calc_avg_mut_loss(pred_avg_mut, avg_mut, criterion)

        self.total_loss = loss + self.lambda1 * segment_id_loss + self.lambda2 * avg_mut_loss + construct_loss *self.lambda3

        return {'local_loss': loss_local, 
                'mid_loss' : loss_mid, 
                'distal_loss' : loss_distal, 
                'loss' : loss, 
                'total_loss' : self.total_loss,
                'segment_id_loss' : segment_id_loss, 
                'avg_mut_loss' : avg_mut_loss,
                'discrim_loss' : discrim_loss,
                'construct_loss' : construct_loss}


    def calc_main_loss(self, predict_out, y, criterion):
        pred_local = predict_out.get('local')
        pred_mid = predict_out.get('mid')
        pred_distal = predict_out.get('distal')
        pred_total = predict_out.get('out')

        loss_local = criterion(pred_local, y)
        loss_mid = criterion(pred_mid, y)
        loss_distal = criterion(pred_distal, y)
        loss = criterion(pred_total, y)
        
        return loss_local, loss_mid, loss_distal, loss
    
    def calc_segment_id_loss(self, pred_segment_id, segment_id):
        criterion = torch.nn.MSELoss(reduction='sum')
        segment_id_loss = criterion(pred_segment_id, segment_id)
        return segment_id_loss

    def calc_avg_mut_loss(self, pred_avg_mut, avg_mut, criterion):
        avg_mut_loss = criterion(pred_avg_mut, avg_mut)
        return avg_mut_loss
    
    def check_preds(self, preds):
        for pred in preds:
            if not isinstance(pred, dict):
                sys.exit(f"Error: pred should be dict type, but input is {type(pred)}")

    def check_labels(self, labels):
        if not isinstance(labels, dict):
            sys.exit(f"Error: labels should be dict type, but input is {type(labels)}")
    def extract_total_loss(self):
        return self.total_loss

class SoftLabelUtilAvgSegmentWithGANLossStrategy():
    def __init__(self) -> None:
        self.total_loss = None
        self.alpha = 0.5
        print("Use Segment Mut as Soft Label, alpha: ", self.alpha)
        self.lambda3 = 1
        self.criterion = nn.KLDivLoss(reduction='sum')
    


    def calc_loss(self, preds, labels, criterion):

        self.check_preds(preds)
        self.check_labels(labels)

        construct_loss = preds.get('construct_loss')

        y = labels.get('label')
        loss_local, loss_mid, loss_distal, hard_loss = self.calc_hard_loss(preds, y, criterion)

        y_one_hot = to_one_hot(y, num_classes=4)
        avg_mut = labels.get('avg_mut')
        mix_label = self.combin_soft_hard_label(hard_label=y_one_hot, soft_label=avg_mut)

        loss = self.calc_main_loss(preds, mix_label)

        if construct_loss is not None:
            self.total_loss = loss + construct_loss *self.lambda3
        else:
            self.total_loss = loss

        return {'local_loss': loss_local, 
                'mid_loss' : loss_mid, 
                'distal_loss' : loss_distal, 
                'loss' : loss, 
                'construct_loss' : construct_loss,
                'hard_label_loss': hard_loss}

    def combin_soft_hard_label(self, hard_label, soft_label):
        mix_label = self.alpha * hard_label + (1-self.alpha) * soft_label
        return mix_label
    
    def calc_hard_loss(self, predict, y, criterion):
        pred_local = predict.get('local')
        pred_mid = predict.get('mid')
        pred_distal = predict.get('distal')
        pred_total = predict.get('out')


        loss_local = criterion(pred_local, y)
        loss_mid = criterion(pred_mid, y)
        loss_distal = criterion(pred_distal, y)
        loss = criterion(pred_total, y)
        return loss_local, loss_mid, loss_distal, loss

    def calc_main_loss(self, predict_out, y):
        pred_total = predict_out.get('out')
        log_probs = nn.functional.log_softmax(pred_total, dim=1)
        loss = self.criterion(log_probs, y)
        return loss
        
    
    
    def check_preds(self, preds):
        if not isinstance(preds, dict):
            sys.exit(f"Error: pred should be dict type, but input is {type(preds)}")

    def check_labels(self, labels):
        if not isinstance(labels, dict):
            sys.exit(f"Error: labels should be dict type, but input is {type(labels)}")
    def extract_total_loss(self):
        return self.total_loss


def to_one_hot(labels, num_classes):
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    return one_hot

class SoftLabelUtilAvgSegmentWithGANLossStrategy2():
    """
    SoftLabelUtilAvgSegmentWithGANLossStrategy2:
        Loss = alpha * soft_loss + (1-alpha) * hard_loss + construct_loss * lambda3
        soft_loss = KLDivLoss(soft_label, y)
        hard_loss = criterion(hard_label, y)
        alpha = 0.7 (default)
    or:
        dymatic sclae
    
    SoftLabelUtilAvgSegmentWithGANLossStrategy:
        Loss = KLDivLoss(alpha * soft_label +  (1-alpha) * hard_label, y) + construct_loss * lambda3
        alpha = 0.5 (default)
    """
    def __init__(self) -> None:
        self.total_loss = None
        #self.dymetic_scale = True
        self.dymetic_scale = False
        self.lambda3 = 1
        self.soft_criterion = nn.KLDivLoss(reduction='sum')

        if self.dymetic_scale:
            self.alpha = 0.5
            #self.calc_mix_loss = self.calc_mix_loss_dymetics_scale
            #print("Use Segment Mut as Soft Label, alpha: ", self.alpha, "; Dynamic Scale: ", self.dymetic_scale)
            self.calc_mix_loss = self.calc_mix_loss_dymetics_scale2
            print("Use Segment Mut as Soft Label, alpha: ", self.alpha, "; Dynamic Scale 2: ", self.dymetic_scale)
        else:
            self.alpha = 0.97
            self.calc_mix_loss = self.calc_mix_loss_abs_scale
            print("Use Segment Mut as Soft Label, alpha: ", self.alpha, "; Dynamic Scale: ", self.dymetic_scale)

    def calc_mix_loss_dymetics_scale(self, soft_loss, hard_loss, alpha):
        soft_weight = hard_loss / (soft_loss + 1e-8)

        mix_loss = alpha * soft_weight * soft_loss + (1 - alpha)* hard_loss 
        return mix_loss
    
    def calc_mix_loss_dymetics_scale2(self, soft_loss, hard_loss, alpha):
        hard_weight = soft_loss / (soft_loss + hard_loss)
        soft_weight = hard / (soft_loss + hard_loss)
        mix_loss = alpha * soft_weight * soft_loss +  (1-alpha)* hard_weight * hard_loss
        return mix_loss
    
    def calc_mix_loss_abs_scale(self, soft_loss, hard_loss, alpha):
        """
        30 muti, alpha should be 0.97
        """
        mix_loss = alpha * soft_loss +  (1 - self.alpha) * hard_loss 

        return mix_loss


    def calc_loss(self, preds, labels, criterion):

        self.check_preds(preds)
        self.check_labels(labels)

        construct_loss = preds.get('construct_loss')

        y = labels.get('label')
        loss_local, loss_mid, loss_distal, loss_hard = self.calc_hard_loss(preds, y, criterion)

        avg_mut = labels.get('avg_mut')

        loss_soft = self.calc_soft_loss(preds, avg_mut)
        loss_mix = self.calc_mix_loss(loss_soft, loss_hard, self.alpha)
        if construct_loss is not None:
            self.total_loss = loss_mix + construct_loss *self.lambda3
        else:
            self.total_loss = loss_mix
        total_loss = self.total_loss

        return {'local_loss': loss_local, 
                'mid_loss' : loss_mid, 
                'distal_loss' : loss_distal, 
                'loss' : total_loss, 
                'construct_loss' : construct_loss,
                'hard_label_loss': loss_hard,
                'soft_label_loss': loss_soft,
                'mix_loss': loss_mix}

    def calc_hard_loss(self, predict, y, criterion):
        pred_local = predict.get('local')
        pred_mid = predict.get('mid')
        pred_distal = predict.get('distal')
        pred_total = predict.get('out')

        loss_local = criterion(pred_local, y)
        loss_mid = criterion(pred_mid, y)
        loss_distal = criterion(pred_distal, y)
        loss = criterion(pred_total, y)
        return loss_local, loss_mid, loss_distal, loss

    def calc_soft_loss(self, predict_out, y):
        pred_total = predict_out.get('out')
        log_probs = nn.functional.log_softmax(pred_total, dim=1)
        loss = self.soft_criterion(log_probs, y)
        return loss
        
    
    
    def check_preds(self, preds):
        if not isinstance(preds, dict):
            sys.exit(f"Error: pred should be dict type, but input is {type(preds)}")

    def check_labels(self, labels):
        if not isinstance(labels, dict):
            sys.exit(f"Error: labels should be dict type, but input is {type(labels)}")
    def extract_total_loss(self):
        return self.total_loss


