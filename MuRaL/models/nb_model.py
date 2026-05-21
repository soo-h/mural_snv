import sys

_ext_path = '/public/home/songhui/project/Mural/Mural_repo/MuRaL_112/model_utils/'
if _ext_path not in sys.path:
    sys.path.append(_ext_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_models_use_segmentInfo import MuRaL_Network3_addfc3
from model_fusion_arg import Network3_ARG_condition


class MuRaL_Network3_addfc3_NB(MuRaL_Network3_addfc3):
    """Model 127 variant with NB loss r prediction head.

    Same architecture as MuRaL_Network3_addfc3, with an additional
    r_head MLP that predicts log_r from fused component probabilities,
    and mu/r activation for Negative Binomial loss.
    """
    def __init__(self, *args, r_head_hidden=16, **kwargs):
        super().__init__(*args, **kwargs)
        n_class = kwargs.get('n_class', 4)
        config = kwargs.get('config', {}) or {}
        fused_type = config.get('fused_type', 'prob')
        self.mu_activation = 'exp' if fused_type == 'prob' else 'softplus'
        self.r_head = nn.Sequential(
            nn.Linear(n_class, r_head_hidden),
            nn.ReLU(),
            nn.Linear(r_head_hidden, n_class),
        )

    def forward(self, local_input, distal_input):
        predict_out, segment_pred = super().forward(local_input, distal_input)

        # Compute fused probability from component logits for r_head
        local = F.softmax(predict_out['local'], dim=1)
        local2 = predict_out.get('local2')
        local3 = predict_out.get('local3')
        mid = F.softmax(predict_out['mid'], dim=1)
        distal = F.softmax(predict_out['distal'], dim=1)
        distal_out = (mid + distal) / 2

        components = [local]
        if local2 is not None:
            components.append(F.softmax(local2, dim=1))
        if local3 is not None:
            components.append(F.softmax(local3, dim=1))
        components.append(distal_out)

        fused_prob = sum(components) / len(components)
        fused_prob = torch.clamp(fused_prob, min=1e-9, max=1.0)

        # r activation: r_head outputs raw logits → softplus → positive r
        log_r = self.r_head(fused_prob)
        r = F.softplus(log_r) + 1e-6

        # mu activation: out format depends on fused_type
        out = predict_out['out']
        if self.mu_activation == 'exp':
            mu = out.exp()
        else:
            mu = F.softplus(out) + 1e-6

        predict_out['r'] = r
        predict_out['mu'] = mu

        return predict_out, segment_pred


class Network3_ARG_condition_NB(Network3_ARG_condition):
    """Model 151 variant with NB loss r prediction head.

    Same architecture as Network3_ARG_condition, with an additional
    r_head MLP that predicts log_r from fused component probabilities,
    and mu/r activation for Negative Binomial loss.

    Note: condition_arg always outputs raw logits, so mu_activation='softplus'.
    """
    def __init__(self, *args, r_head_hidden=16, **kwargs):
        super().__init__(*args, **kwargs)
        n_class = kwargs.get('n_class', 4)
        self.mu_activation = 'softplus'  # condition_arg always outputs raw logits
        self.r_head = nn.Sequential(
            nn.Linear(n_class, r_head_hidden),
            nn.ReLU(),
            nn.Linear(r_head_hidden, n_class),
        )

    def forward(self, local_input, distal_input, arg_feature):
        predict_out, segment_pred = super().forward(local_input, distal_input, arg_feature)

        # Compute fused probability from component logits for r_head
        local = F.softmax(predict_out['local'], dim=1)
        local2 = predict_out.get('local2')
        local3 = predict_out.get('local3')
        mid = F.softmax(predict_out['mid'], dim=1)
        distal = F.softmax(predict_out['distal'], dim=1)
        distal_out = (mid + distal) / 2

        components = [local]
        if local2 is not None:
            components.append(F.softmax(local2, dim=1))
        if local3 is not None:
            components.append(F.softmax(local3, dim=1))
        components.append(distal_out)

        fused_prob = sum(components) / len(components)
        fused_prob = torch.clamp(fused_prob, min=1e-9, max=1.0)

        # r activation: r_head outputs raw logits → softplus → positive r
        log_r = self.r_head(fused_prob)
        r = F.softplus(log_r) + 1e-6

        # mu activation: condition_arg always outputs raw logits → softplus
        out = predict_out['out']
        mu = F.softplus(out) + 1e-6

        predict_out['r'] = r
        predict_out['mu'] = mu

        return predict_out, segment_pred
