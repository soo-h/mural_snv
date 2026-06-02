"""
Dirichlet MDN models.

Extends model 151 (Network3_ARG_condition) with a Dirichlet Mixture
Density Network head that outputs pi_logits and alpha_raw for
Dirichlet MDN classification loss.
"""

import sys

_ext_path = '/public/home/songhui/project/Mural/Mural_repo/MuRaL_112/model_utils/'
if _ext_path not in sys.path:
    sys.path.append(_ext_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_fusion_arg import Network3_ARG_condition
from MuRaL.models.nn_models import Network3


class DirichletMDNHead(nn.Module):
    """Dirichlet Mixture Density Network head.

    Architecture: in_dim -> shared MLP -> pi_head / alpha_head.
    Outputs raw values (no activation). Activation is handled by
    the loss function and predict_from_output independently.

    Args:
        in_dim: Input feature dimension
        K: Number of Dirichlet mixture components
        C: Number of classes (default 4)
        hidden_dim: Shared hidden layer dimension (default 64)
    """
    def __init__(self, in_dim, K=3, C=4, hidden_dim=64):
        super().__init__()
        self.K = K
        self.C = C

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden_dim, K)          # -> pi_logits (raw)
        self.alpha = nn.Linear(hidden_dim, K * C)   # -> alpha_raw (raw)

    def forward(self, x):
        h = self.backbone(x)                                  # [batch, hidden_dim]
        pi_logits = self.pi(h)                                # [batch, K]
        alpha_raw = self.alpha(h).view(-1, self.K, self.C)    # [batch, K, C]
        return pi_logits, alpha_raw


def dirichlet_mdn_predict_from_output(out, eps=1e-8):
    """Infer final predictions from Dirichlet MDN model output.

    out should contain raw 'pi_logits' and 'alpha_raw'.
    Activation (softmax, softplus) is applied internally.

    Args:
        out: dict with keys 'pi_logits' and 'alpha_raw'
    Returns:
        dict with prob, logits, pred_class, pi, alpha, p_k, evidence
    """
    pi_logits = out['pi_logits']       # (B, K), raw
    alpha_raw = out['alpha_raw']       # (B, K, C), raw

    pi = F.softmax(pi_logits, dim=1)                               # (B, K)
    alpha = F.softplus(alpha_raw) + eps                            # (B, K, C)

    p_k = alpha / alpha.sum(dim=-1, keepdim=True)                  # (B, K, C)
    prob = (pi.unsqueeze(-1) * p_k).sum(dim=1)                     # (B, C)

    evidence_k = alpha.sum(dim=-1)                                 # (B, K)
    evidence = (pi * evidence_k).sum(dim=1)                        # (B,)

    return {
        'prob': prob,
        'logits': torch.log(prob + eps),
        'pred_class': prob.argmax(dim=-1),
        'pi': pi,
        'alpha': alpha,
        'p_k': p_k,
        'evidence': evidence,
    }


class Network3_ARG_condition_DirMDN(Network3_ARG_condition):
    """Model 151 variant with Dirichlet MDN head.

    Key differences from base Network3_ARG_condition:
      - condition_arg is truncated at the 128-dim hidden layer
      - 128-dim hidden is fed into DirichletMDNHead (outputs raw values)
      - predict_out['out'] is inferred from pi_logits/alpha_raw
      - Adds predict_out['pi_logits'], predict_out['alpha_raw']
    """

    def __init__(self, *args, K=3, **kwargs):
        super().__init__(*args, **kwargs)

        # Split condition_arg: keep only the projection to 128-dim
        self.condition_arg_proj = self.condition_arg[:3]  # Linear(68,128)->ReLU->Dropout
        del self.condition_arg

        # Dirichlet MDN Head (outputs raw values)
        self.mdn_head = DirichletMDNHead(
            in_dim=128, K=K, C=kwargs.get('n_class', 4),
        )

    def forward(self, local_input, distal_input, arg_feature):
        # --- Reuse base model computation up to fusion ---
        local_outs = self.local_scale_model(local_input)
        local_out = local_outs.get('local_out')
        local_out2 = local_outs.get('local_out2')
        local_out3 = local_outs.get('local_out3')

        if self.middle_radius is not None:
            distal_out1 = self.middle_scale_model(distal_input, self.middle_radius)
        else:
            distal_out1 = self.middle_scale_model(distal_input)

        out_dict = self.large_scale_model(distal_input)
        distal_out2 = out_dict.get('main_pred') if isinstance(out_dict, dict) else out_dict
        distal_out = (distal_out1 + distal_out2) / 2
        arg_feature = self.arg_branch(arg_feature)

        predict_out = {
            'local': local_out,
            'local2': local_out2,
            'local3': local_out3,
            'mid': distal_out1,
            'distal': distal_out2,
        }

        fusion_out = self.fusion(local_out, local_out2, local_out3, distal_out)
        hidden_128 = self.condition_arg_proj(
            torch.concat([fusion_out, arg_feature], dim=1)
        )  # [batch, 128]

        # MDN head outputs raw values
        pi_logits, alpha_raw = self.mdn_head(hidden_128)
        predict_out['pi_logits'] = pi_logits   # [batch, K], raw
        predict_out['alpha_raw'] = alpha_raw   # [batch, K, C], raw

        # out inferred from pi_logits/alpha_raw (for eval compatibility)
        inferred = dirichlet_mdn_predict_from_output(predict_out)
        predict_out['out'] = inferred['logits']

        if 'local_h1' in local_outs:
            assert 'local_h2' in local_outs
            predict_out['local_h1'] = local_outs['local_h1']
            predict_out['local_h2'] = local_outs['local_h2']

        return predict_out, None


class Network3_DirMDN(Network3):
    """Model3 variant with Dirichlet MDN head.

    Uses submodel output logits concatenated as input to DirMDN head.
    """

    def __init__(self, *args, K=3, **kwargs):
        super().__init__(*args, **kwargs)

        # concat_dim = submodel_count * n_class (4->16 or 3->12)
        submodel_count = 4 if self.no_of_cont > 0 else 3
        proj_in_dim = submodel_count * self.n_class

        self.dir_mdn_proj = nn.Sequential(
            nn.Linear(proj_in_dim, 64),
            nn.ReLU(),
        )
        self.mdn_head = DirichletMDNHead(
            in_dim=64, K=K, C=self.n_class, hidden_dim=64,
        )
        self.K = K

    def forward(self, local_input, distal_input):
        # Step 1: Reuse base forward
        base_out, _ = super().forward(local_input, distal_input)

        # Step 2: Concat submodel logits (fixed order)
        logit_list = [
            base_out['local'],
            base_out['mid'],
            base_out['distal'],
        ]
        if base_out.get('local2') is not None:
            logit_list.append(base_out['local2'])

        concat = torch.cat(logit_list, dim=1)    # [B, 12~16]
        hidden = self.dir_mdn_proj(concat)       # [B, 64]

        pi_logits, alpha_raw = self.mdn_head(hidden)

        # Step 3: Build output dict
        base_out['pi_logits'] = pi_logits         # [B, K], raw
        base_out['alpha_raw'] = alpha_raw         # [B, K, C], raw

        inferred = dirichlet_mdn_predict_from_output(base_out)
        base_out['out'] = inferred['logits']

        return base_out, None
