"""
Gamma MDN models.

Extends model 151 (Network3_ARG_condition) with a Gamma Mixture
Density Network head that outputs pi_logits, alpha_raw, and beta_raw for
Gamma MDN classification loss.
"""

import sys

_ext_path = '/public/home/songhui/project/Mural/Mural_repo/MuRaL_112/model_utils/'
if _ext_path not in sys.path:
    sys.path.append(_ext_path)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_fusion_arg import Network3_ARG_condition


class GammaMDNHead(nn.Module):
    """Gamma Mixture Density Network head.

    Architecture: in_dim -> shared MLP -> pi_head / alpha_head / beta_head.
    Outputs raw values (no activation). Activation is handled by
    the loss function and predict_from_output independently.

    Args:
        in_dim: Input feature dimension
        K: Number of Gamma mixture components
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
        self.pi = nn.Linear(hidden_dim, K)            # -> pi_logits (raw)
        self.alpha = nn.Linear(hidden_dim, K * C)     # -> alpha_raw (raw)
        self.beta = nn.Linear(hidden_dim, K * C)      # -> beta_raw (raw)

    def forward(self, x):
        h = self.backbone(x)                                        # [batch, hidden_dim]
        pi_logits = self.pi(h)                                      # [batch, K]
        alpha_raw = self.alpha(h).view(-1, self.K, self.C)          # [batch, K, C]
        beta_raw = self.beta(h).view(-1, self.K, self.C)            # [batch, K, C]
        return pi_logits, alpha_raw, beta_raw


def gamma_mdn_predict_from_output(out, eps=1e-8):
    """Infer final predictions from Gamma MDN model output.

    out should contain raw 'pi_logits', 'alpha_raw', 'beta_raw'.
    Activation (softmax, softplus) is applied internally.
    Only returns prediction-related quantities; see compute_mdn_uncertainty
    for uncertainty metrics.

    Args:
        out: dict with keys 'pi_logits', 'alpha_raw', 'beta_raw'
    Returns:
        dict with prob, logits, pred_class
    """
    pi_logits = out['pi_logits']       # (B, K), raw
    alpha_raw = out['alpha_raw']       # (B, K, C), raw
    beta_raw = out['beta_raw']         # (B, K, C), raw

    pi = F.softmax(pi_logits, dim=1)                               # (B, K)
    alpha = F.softplus(alpha_raw) + eps                            # (B, K, C)
    beta = F.softplus(beta_raw) + eps                              # (B, K, C)

    lam = alpha / beta                                              # (B, K, C), Gamma mean
    p_k = lam / lam.sum(dim=-1, keepdim=True)                      # (B, K, C)
    prob = (pi.unsqueeze(-1) * p_k).sum(dim=1)                     # (B, C)

    return {
        'prob': prob,
        'logits': torch.log(prob + eps),
        'pred_class': prob.argmax(dim=-1),
    }


def compute_mdn_uncertainty(predict_out, eps=1e-8):
    """Compute uncertainty metrics from MDN model output.

    Takes model forward's predict_out dict (contains raw values),
    applies activation internally. Currently only pi_entropy.
    Caller controls gradient tracking (e.g. torch.no_grad()).

    Args:
        predict_out: dict, must contain 'pi_logits' (B, K)
    Returns:
        dict with 'pi_entropy': (B,) Tensor
    """
    pi_logits = predict_out['pi_logits']          # (B, K), raw

    pi = F.softmax(pi_logits, dim=1)                           # (B, K)
    pi_entropy = -(pi * torch.log(pi + eps)).sum(dim=1)        # (B,)

    return {
        'pi_entropy': pi_entropy,
    }


class Network3_ARG_condition_GammaMDN(Network3_ARG_condition):
    """Model 151 variant with Gamma MDN head.

    Key differences from base Network3_ARG_condition:
      - condition_arg is truncated at the 128-dim hidden layer
      - 128-dim hidden is fed into GammaMDNHead (outputs raw values)
      - predict_out['out'] is inferred from pi_logits/alpha_raw/beta_raw
      - Adds predict_out['pi_logits'], predict_out['alpha_raw'], predict_out['beta_raw']
    """

    def __init__(self, *args, K=3, **kwargs):
        super().__init__(*args, **kwargs)

        # Split condition_arg: keep only the projection to 128-dim
        self.condition_arg_proj = self.condition_arg[:3]  # Linear(68,128)->ReLU->Dropout
        del self.condition_arg

        # Gamma MDN Head (outputs raw values)
        self.mdn_head = GammaMDNHead(
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
        pi_logits, alpha_raw, beta_raw = self.mdn_head(hidden_128)
        predict_out['pi_logits'] = pi_logits   # [batch, K], raw
        predict_out['alpha_raw'] = alpha_raw   # [batch, K, C], raw
        predict_out['beta_raw'] = beta_raw     # [batch, K, C], raw

        # out inferred from pi_logits/alpha_raw/beta_raw (for eval compatibility)
        inferred = gamma_mdn_predict_from_output(predict_out)
        predict_out['out'] = inferred['logits']

        if 'local_h1' in local_outs:
            predict_out['local_h1'] = local_outs['local_h1']
            predict_out['local_h2'] = local_outs['local_h2']

        return predict_out, None
