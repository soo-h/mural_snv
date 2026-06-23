"""Calibration, loss metrics, and evaluation functions.

Calibration wrappers (jax + dirichletcal), ECE/Brier/Focal/CB losses,
and re-exports of gradient/metric utilities from sub-modules.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export sub-module utilities so ``from MuRaL.evaluation.evaluation import *``
# continues to work for all existing callers.
from MuRaL.evaluation.gradient_utils import (  # noqa: F401
    count_parameters, check_gradients, print_gradient_norms,
    print_gradients, hook_backward_function,
)
from MuRaL.evaluation.metrics import (  # noqa: F401
    f3mer_comp, freq_kmer_comp_multi, corr_calc_sub, calc_avg_prob,
)

# Import warnings filter
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import jax
jax.config.update('jax_platform_name', 'cpu')

from dirichletcal.calib.vectorscaling import VectorScaling
from dirichletcal.calib.tempscaling import TemperatureScaling
from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator


class ECELoss(nn.Module):
    """Compute ECE (Expected Calibration Error).

    Use code from https://github.com/torrvision/focal_calibration
    """
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ClasswiseECELoss(nn.Module):
    """Compute Classwise ECE.

    Use code from https://github.com/torrvision/focal_calibration
    """
    def __init__(self, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i)

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if i == 0:
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce


class BrierScore(nn.Module):
    """Implementation of Brier score."""
    def __init__(self):
        super(BrierScore, self).__init__()

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        target_one_hot = torch.FloatTensor(input.shape).to(target.device)
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = F.softmax(input, dim=1)
        squared_diff = (target_one_hot - pt) ** 2
        loss = torch.sum(squared_diff) / float(input.shape[0])
        return loss


def calibrate_prob(y_prob, y, device, calibr_name='FullDiri'):
    """Fit a calibrator from the dirichletcal package.

    Use calibrators in dirichletcal package.
    """
    if calibr_name == 'VectS':
        calibr = VectorScaling(logit_constant=0.0)
    elif calibr_name == 'TempS':
        calibr = TemperatureScaling(logit_constant=0.0)
    elif calibr_name == 'FullDiri':
        calibr = FullDirichletCalibrator()
    elif calibr_name == 'FullDiriODIR':
        l2_odir = 1e-2
        calibr = FullDirichletCalibrator(reg_lambda=l2_odir, reg_mu=l2_odir)
    elif calibr_name == 'FullDiri1':
        calibr = FullDirichletCalibrator(reg_norm=True)
    elif calibr_name == 'FullDiri2':
        calibr = FullDirichletCalibrator(ref_row=False)

    calibr.fit(y_prob, y)
    prob_cal = calibr.predict_proba(y_prob)
    print('y_prob.head():', y_prob[0:6, ])
    print('y:', y[0:6])
    print('prob_cal:', prob_cal[0:6, ])
    print('calibr.coef_: ', calibr.coef_)
    print('calibr.weights_:', calibr.weights_)
    print("prob_cal.min:", prob_cal.min(axis=0))
    print("prob_cal.max:", prob_cal.max(axis=0))
    print("CV:", y_prob.std(axis=0) / y_prob.mean(axis=0))
    print("CV (after calibration):", prob_cal.std(axis=0) / prob_cal.mean(axis=0))

    nll_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    ece_criterion = ECELoss(n_bins=50).to(device)
    c_ece_criterion = ClasswiseECELoss(n_bins=50).to(device)
    brier_criterion = BrierScore().to(device)

    logits0 = torch.log(torch.from_numpy(y_prob)).to(device)
    logits = torch.log(torch.from_numpy(np.copy(prob_cal))).to(device)
    labels = torch.from_numpy(y).long().to(device)

    nll0 = nll_criterion(logits0, labels).item()
    nll = nll_criterion(logits, labels).item()
    ece0 = ece_criterion(logits0, labels).item()
    ece = ece_criterion(logits, labels).item()
    c_ece0 = c_ece_criterion(logits0, labels).item()
    c_ece = c_ece_criterion(logits, labels).item()
    brier0 = brier_criterion(logits0, labels).item()
    brier = brier_criterion(logits, labels).item()

    print('Before ' + calibr_name + ' scaling - NLL: %.8f, ECE: %.8f, CwECE: %.8f, Brier: %.8f'
          % (nll0, ece0, c_ece0, brier0))
    print('After ' + calibr_name + ' scaling - NLL: %.8f, ECE: %.8f, CwECE: %.8f, Brier: %.8f'
          % (nll, ece, c_ece, brier))

    return calibr, nll


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class CBLoss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, loss_type="sigmoid", beta=0.9999, gamma=1):
        super(CBLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, labels):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.to(logits.device)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = self._focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss

    @staticmethod
    def _focal_loss(labels, logits, alpha, gamma):
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")
        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                  torch.exp(-1.0 * logits)))
        loss = modulator * BCLoss
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(labels)
        return focal_loss
