import sys

# 确保可以 import 外部 base class
_ext_path = '/public/home/songhui/project/Mural/Mural_repo/MuRaL_112/model_utils/'
if _ext_path not in sys.path:
    sys.path.append(_ext_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_models_use_segmentInfo import MuRaL_Network3_addfc3
from model_fusion_arg import Network3_ARG_condition


class MuRaL_Network3_addfc3_NBv3(MuRaL_Network3_addfc3):
    """Model 127 variant with component-level high-dim feature r heads (v3).

    Unlike v1 (fused prob -> MLP) and v2 (per-component prob -> Linear),
    v3 captures the intermediate high-dimensional features before the final
    projection to n_class in each sub-model, using forward hooks.

    Hooks target the last Linear layer in each sub-model's final projection
    module, capturing the feature right before classification:

      - local  (local_fc input):       dim=lin_layer_sizes[-1] -> MLP(->32->ReLU->n_class)
      - local2 (local_fc2 last Linear): dim=varies              -> MLP(->16->ReLU->n_class)
      - local3 (local_model3 last Lin): dim=varies              -> MLP(->16->ReLU->n_class)
      - mid    (distal_fc1 last Lin):   dim=out_channels        -> MLP(->16->ReLU->n_class)
      - distal (distal_fc2 last Lin):   dim=cnn_fc_in_size      -> MLP(->16->ReLU->n_class)

    Fusion follows the same weighting scheme as `out`:
      distal_r_fused = (mid_r + distal_r) / 2
      log_r = avg(local_r, local2_r, local3_r, distal_r_fused)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_class = kwargs.get('n_class', 4)
        config = kwargs.get('config', {}) or {}
        fused_type = config.get('fused_type', 'prob')
        self.mu_activation = 'exp' if fused_type == 'prob' else 'softplus'

        # Find the last Linear layer in each sub-model's final projection
        local_linear = self._find_last_linear(self.local_scale_model.local_fc)
        local_dim = local_linear.in_features

        self._has_local2 = config.get('use_local_fc2', True)
        self._has_local3 = config.get('use_local_fc3', True)

        mid_linear = self._find_last_linear(self.middle_scale_model.distal_fc1)
        mid_dim = mid_linear.in_features

        distal_linear = self._find_last_linear(self.large_scale_model.distal_fc2)
        distal_dim = distal_linear.in_features

        # Per-component MLPs on high-dim features
        self.r_head_local = nn.Sequential(
            nn.Linear(local_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_class),
        )
        self.r_head_mid = nn.Sequential(
            nn.Linear(mid_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_class),
        )
        self.r_head_distal = nn.Sequential(
            nn.Linear(distal_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_class),
        )

        if self._has_local2:
            local2_linear = self._find_last_linear(self.local_scale_model.local_fc2)
            local2_dim = local2_linear.in_features
            self.r_head_local2 = nn.Sequential(
                nn.Linear(local2_dim, 16),
                nn.ReLU(),
                nn.Linear(16, n_class),
            )

        if self._has_local3:
            local3_linear = self._find_last_linear(self.local_scale_model.local_model3)
            local3_dim = local3_linear.in_features
            self.r_head_local3 = nn.Sequential(
                nn.Linear(local3_dim, 16),
                nn.ReLU(),
                nn.Linear(16, n_class),
            )

        # Forward hooks
        self._v3_feat = {}

        def _make_hook(name):
            def hook_fn(module, input_, output_):
                self._v3_feat[name] = input_[0]
            return hook_fn

        self._hook_handles = [
            local_linear.register_forward_hook(_make_hook('local')),
        ]
        if self._has_local2:
            local2_linear = self._find_last_linear(self.local_scale_model.local_fc2)
            self._hook_handles.append(
                local2_linear.register_forward_hook(_make_hook('local2')))
        if self._has_local3:
            local3_linear = self._find_last_linear(self.local_scale_model.local_model3)
            self._hook_handles.append(
                local3_linear.register_forward_hook(_make_hook('local3')))

        self._hook_handles += [
            mid_linear.register_forward_hook(_make_hook('mid')),
            distal_linear.register_forward_hook(_make_hook('distal')),
        ]

    @staticmethod
    def _find_last_linear(module):
        last = None
        for m in module.modules():
            if isinstance(m, nn.Linear):
                last = m
        return last

    def forward(self, local_input, distal_input):
        predict_out, segment_pred = super().forward(local_input, distal_input)

        # Pop captured high-dim features
        local_feat = self._v3_feat.pop('local', None)
        mid_feat = self._v3_feat.pop('mid', None)
        distal_feat = self._v3_feat.pop('distal', None)

        missing = []
        if local_feat is None:
            missing.append('local')
        if mid_feat is None:
            missing.append('mid')
        if distal_feat is None:
            missing.append('distal')

        local2_feat = self._v3_feat.pop('local2', None) if self._has_local2 else None
        local3_feat = self._v3_feat.pop('local3', None) if self._has_local3 else None

        if self._has_local2 and local2_feat is None:
            missing.append('local2')
        if self._has_local3 and local3_feat is None:
            missing.append('local3')

        if missing:
            raise RuntimeError(
                'NBv3: forward hooks failed to capture features: {}. '
                'Check sub-model layer names.'.format(missing))

        # Per-component log_r from high-dim features
        local_r = self.r_head_local(local_feat)
        mid_r = self.r_head_mid(mid_feat)
        distal_r = self.r_head_distal(distal_feat)

        distal_r_fused = (mid_r + distal_r) / 2

        components_r = [local_r]
        if self._has_local2:
            components_r.append(self.r_head_local2(local2_feat))
        if self._has_local3:
            components_r.append(self.r_head_local3(local3_feat))
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

    def _cleanup_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []


class Network3_ARG_condition_NBv3(Network3_ARG_condition):
    """Model 151 variant with component-level high-dim feature r heads (v3).

    Same v3 design as MuRaL_Network3_addfc3_NBv3: hooks on the last Linear
    layer of each sub-model (local, local2, local3, mid, distal) capture
    high-dimensional features before classification; per-component MLPs
    predict log_r; fusion mirrors the base model's out weighting.

    Note: condition_arg always outputs raw logits, so mu_activation='softplus'.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_class = kwargs.get('n_class', 4)
        config = kwargs.get('config', {}) or {}
        self.mu_activation = 'softplus'  # condition_arg always outputs raw logits

        local_linear = self._find_last_linear(self.local_scale_model.local_fc)
        local_dim = local_linear.in_features

        self._has_local2 = config.get('use_local_fc2', True)
        self._has_local3 = config.get('use_local_fc3', True)

        mid_linear = self._find_last_linear(self.middle_scale_model.distal_fc1)
        mid_dim = mid_linear.in_features

        distal_linear = self._find_last_linear(self.large_scale_model.distal_fc2)
        distal_dim = distal_linear.in_features

        self.r_head_local = nn.Sequential(
            nn.Linear(local_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_class),
        )
        self.r_head_mid = nn.Sequential(
            nn.Linear(mid_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_class),
        )
        self.r_head_distal = nn.Sequential(
            nn.Linear(distal_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_class),
        )

        if self._has_local2:
            local2_linear = self._find_last_linear(self.local_scale_model.local_fc2)
            local2_dim = local2_linear.in_features
            self.r_head_local2 = nn.Sequential(
                nn.Linear(local2_dim, 16),
                nn.ReLU(),
                nn.Linear(16, n_class),
            )

        if self._has_local3:
            local3_linear = self._find_last_linear(self.local_scale_model.local_model3)
            local3_dim = local3_linear.in_features
            self.r_head_local3 = nn.Sequential(
                nn.Linear(local3_dim, 16),
                nn.ReLU(),
                nn.Linear(16, n_class),
            )

        self._v3_feat = {}

        def _make_hook(name):
            def hook_fn(module, input_, output_):
                self._v3_feat[name] = input_[0]
            return hook_fn

        self._hook_handles = [
            local_linear.register_forward_hook(_make_hook('local')),
        ]
        if self._has_local2:
            local2_linear = self._find_last_linear(self.local_scale_model.local_fc2)
            self._hook_handles.append(
                local2_linear.register_forward_hook(_make_hook('local2')))
        if self._has_local3:
            local3_linear = self._find_last_linear(self.local_scale_model.local_model3)
            self._hook_handles.append(
                local3_linear.register_forward_hook(_make_hook('local3')))

        self._hook_handles += [
            mid_linear.register_forward_hook(_make_hook('mid')),
            distal_linear.register_forward_hook(_make_hook('distal')),
        ]

    @staticmethod
    def _find_last_linear(module):
        last = None
        for m in module.modules():
            if isinstance(m, nn.Linear):
                last = m
        return last

    def forward(self, local_input, distal_input, arg_feature):
        predict_out, segment_pred = super().forward(local_input, distal_input, arg_feature)

        local_feat = self._v3_feat.pop('local', None)
        mid_feat = self._v3_feat.pop('mid', None)
        distal_feat = self._v3_feat.pop('distal', None)

        missing = []
        if local_feat is None:
            missing.append('local')
        if mid_feat is None:
            missing.append('mid')
        if distal_feat is None:
            missing.append('distal')

        local2_feat = self._v3_feat.pop('local2', None) if self._has_local2 else None
        local3_feat = self._v3_feat.pop('local3', None) if self._has_local3 else None

        if self._has_local2 and local2_feat is None:
            missing.append('local2')
        if self._has_local3 and local3_feat is None:
            missing.append('local3')

        if missing:
            raise RuntimeError(
                'NBv3 (151): forward hooks failed to capture features: {}. '
                'Check sub-model layer names.'.format(missing))

        local_r = self.r_head_local(local_feat)
        mid_r = self.r_head_mid(mid_feat)
        distal_r = self.r_head_distal(distal_feat)

        distal_r_fused = (mid_r + distal_r) / 2

        components_r = [local_r]
        if self._has_local2:
            components_r.append(self.r_head_local2(local2_feat))
        if self._has_local3:
            components_r.append(self.r_head_local3(local3_feat))
        components_r.append(distal_r_fused)

        log_r = sum(components_r) / len(components_r)
        r = F.softplus(log_r) + 1e-6

        # mu activation: condition_arg always outputs raw logits → softplus
        out = predict_out['out']
        mu = F.softplus(out) + 1e-6

        predict_out['r'] = r
        predict_out['mu'] = mu

        return predict_out, segment_pred

    def _cleanup_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
