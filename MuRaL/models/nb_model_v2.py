"""
NBv2 models for NB loss.

Model 127 (MuRaL_Network3_addfc3_NBv2):
  Per-component prob -> Linear(n_class, n_class) -> component_log_r -> average
  Same as before, unchanged.

Model 151 (Network3_ARG_condition_NBv2):
  r_head takes concat([out, prob_var]) -> 3-layer MLP -> log_r.
  prob_var captures sub-model disagreement; out brings condition_arg signal.
"""

import sys

_ext_path = '/public/home/songhui/project/Mural/Mural_repo/MuRaL_112/model_utils/'
if _ext_path not in sys.path:
    sys.path.append(_ext_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_models_use_segmentInfo import MuRaL_Network3_addfc3
from model_fusion_arg import Network3_ARG_condition


class MuRaL_Network3_addfc3_NBv2(MuRaL_Network3_addfc3):
    """Model 127 variant with component-level r heads for NB loss.

    Unlike MuRaL_Network3_addfc3_NB which computes log_r from fused prob,
    each component (local, local2, local3, mid, distal) independently
    predicts log_r via its own Linear head, then fused with the same
    weighting scheme as the out probability. This preserves sub-model
    disagreement information for dispersion estimation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_class = kwargs.get('n_class', 4)
        config = kwargs.get('config', {}) or {}
        fused_type = config.get('fused_type', 'prob')
        self.mu_activation = 'exp' if fused_type == 'prob' else 'softplus'
        self.r_head_local = nn.Linear(n_class, n_class)
        self.r_head_local2 = nn.Linear(n_class, n_class)
        self.r_head_local3 = nn.Linear(n_class, n_class)
        self.r_head_mid = nn.Linear(n_class, n_class)
        self.r_head_distal = nn.Linear(n_class, n_class)

    def forward(self, local_input, distal_input):
        predict_out, segment_pred = super().forward(local_input, distal_input)

        # Component logits → softmax → Linear → r fusion
        local = predict_out['local']
        local2 = predict_out.get('local2')
        local3 = predict_out.get('local3')
        mid = predict_out['mid']
        distal = predict_out['distal']

        local_r = self.r_head_local(F.softmax(local, dim=1))
        local2_r = self.r_head_local2(F.softmax(local2, dim=1)) if local2 is not None else None
        local3_r = self.r_head_local3(F.softmax(local3, dim=1)) if local3 is not None else None
        mid_r = self.r_head_mid(F.softmax(mid, dim=1))
        distal_r = self.r_head_distal(F.softmax(distal, dim=1))

        # Fuse log_r with same weighting as out probability
        distal_r_fused = (mid_r + distal_r) / 2
        components_r = [local_r]
        if local2_r is not None:
            components_r.append(local2_r)
        if local3_r is not None:
            components_r.append(local3_r)
        components_r.append(distal_r_fused)

        log_r = sum(components_r) / len(components_r)
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


class Network3_ARG_condition_NBv2(Network3_ARG_condition):
    """Network3_ARG_condition with NBv2 variance-aware r_head.

    Unlike v1 (r from fused prob) and v2 127 (per-component Linear → average),
    v2 151 computes r from the concatenation of:
      - out (raw logits, already incorporates arg_feature via condition_arg)
      - prob_var (variance across sub-model softmax probabilities)

    This allows r to perceive both the arg-conditioned prediction and the
    disagreement among sub-models: high disagreement → lower r (more dispersion).
    """
    def __init__(self,
                 emb_dims,
                 no_of_cont,
                 lin_layer_sizes,
                 emb_dropout,
                 lin_layer_dropouts,
                 in_channels,
                 out_channels,
                 kernel_size,
                 distal_radius,
                 distal_order,
                 distal_fc_dropout,
                 n_class,
                 emb_padding_idx=None,
                 config=None,
                 avgmut_dropout=None):
        super().__init__(
            emb_dims, no_of_cont, lin_layer_sizes, emb_dropout, lin_layer_dropouts,
            in_channels, out_channels, kernel_size, distal_radius, distal_order,
            distal_fc_dropout, n_class, emb_padding_idx=emb_padding_idx,
            config=config, avgmut_dropout=avgmut_dropout,
        )

        if config is None:
            config = {}

        self.mu_activation = 'softplus'  # condition_arg always outputs raw logits
        self._has_local2 = config.get('use_local_fc2', True)
        self._has_local3 = config.get('use_local_fc3', True)

        # r_head: concat([out, prob_var]) → 3-layer MLP → log_r
        self.r_head = nn.Sequential(
            nn.Linear(n_class * 2, 16),    # concat([out, prob_var]) = 8
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, n_class),          # per-class log_r
        )

    def forward(self, local_input, distal_input, arg_feature):
        predict_out, segment_pred = super().forward(local_input, distal_input, arg_feature)

        # Component logits → probabilities
        local_prob = F.softmax(predict_out['local'], dim=1)
        mid_prob = F.softmax(predict_out['mid'], dim=1)
        distal_prob = F.softmax(predict_out['distal'], dim=1)

        # Collect available component probs for variance
        comp_probs = [local_prob]

        local2 = predict_out.get('local2')
        if local2 is not None and self._has_local2:
            comp_probs.append(F.softmax(local2, dim=1))

        local3 = predict_out.get('local3')
        if local3 is not None and self._has_local3:
            comp_probs.append(F.softmax(local3, dim=1))

        # Fuse mid + distal before variance (matches out fusion)
        comp_probs.append((mid_prob + distal_prob) / 2)

        # Variance across components: disagreement signal
        stacked = torch.stack(comp_probs, dim=1)  # [N, n_comp, n_class]
        prob_var = stacked.var(dim=1)              # [N, n_class]

        # r_head input: raw out + prob variance
        out = predict_out['out']                   # [N, n_class]
        r_input = torch.cat([out, prob_var], dim=1)  # [N, 2*n_class]
        log_r = self.r_head(r_input)                # [N, n_class]

        r = F.softplus(log_r) + 1e-6
        mu = F.softplus(out) + 1e-6

        predict_out['r'] = r
        predict_out['mu'] = mu
        predict_out['log_r'] = log_r

        return predict_out, segment_pred
