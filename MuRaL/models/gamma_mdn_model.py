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
        C: Number of classes/mutation types (default 4). When C=3,
            the head outputs only mutation-type parameters (Poisson mode).
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


def poisson_gamma_mdn_predict_from_output(out, eps=1e-8):
    """Infer final predictions from Poisson-Gamma MDN model output.

    Non-mutation probability is derived from Poisson process:
      P(no mutation) = exp(-sum_i λ_i)
      P(mutation type i) = (1 - exp(-λ_total)) * λ_i / λ_total

    This matches the PoissonGammaMDNClassificationLoss derivation.
    """
    pi_logits = out['pi_logits']       # (B, K)
    alpha_raw = out['alpha_raw']       # (B, K, 3)
    beta_raw = out['beta_raw']         # (B, K, 3)

    pi = F.softmax(pi_logits, dim=1)                               # (B, K)
    alpha = F.softplus(alpha_raw) + eps                            # (B, K, 3)
    beta = F.softplus(beta_raw) + eps                              # (B, K, 3)
    lam = alpha / beta                                              # (B, K, 3)
    lam_total = lam.sum(dim=-1)                                     # (B, K)

    p_k0 = torch.exp(-lam_total)                                    # (B, K)
    p_ki = (1 - torch.exp(-lam_total)).unsqueeze(-1) * lam / (lam_total.unsqueeze(-1) + eps)
    p_k = torch.cat([p_k0.unsqueeze(-1), p_ki], dim=-1)            # (B, K, 4)

    prob = (pi.unsqueeze(-1) * p_k).sum(dim=1)                     # (B, 4)

    return {
        'prob': prob,
        'logits': torch.log(prob + eps),
        'pred_class': prob.argmax(dim=-1),
    }


def activate_gamma_alpha_beta(
    alpha_raw,
    beta_raw,
    *,
    alpha_min=0.05,
    beta_min=1e-3,
    log_alpha_min=-10.0,
    log_alpha_max=6.0,
    log_beta_min=-10.0,
    log_beta_max=8.0,
    eps=1e-8,
):
    """Activate raw model outputs into Gamma parameters.

    alpha_raw ≈ log(alpha - alpha_min)
    beta_raw  ≈ log(beta - beta_min)

    Clamp upper bounds prevent exp overflow (x ≥ 88 → inf in float32)
    while keeping the full exp gradient within [exp(lo), exp(hi)].
    Use gradient clipping in the optimizer to contain gradient asymmetry
    across mutation types.
    """
    log_alpha = alpha_raw.clamp(min=log_alpha_min, max=log_alpha_max)
    log_beta = beta_raw.clamp(min=log_beta_min, max=log_beta_max)

    alpha = alpha_min + torch.exp(log_alpha)
    beta = beta_min + torch.exp(log_beta)

    alpha = torch.clamp(alpha, min=eps)
    beta = torch.clamp(beta, min=eps)

    return alpha, beta


def poisson_gamma_log_predict_from_output(out, eps=1e-8,
        log_alpha_max=6.0, log_beta_max=8.0, **act_kwargs):
    """Poisson-Gamma MDN predict with log-parameterization."""
    pi_logits = out['pi_logits']       # (B, K)
    alpha_raw = out['alpha_raw']       # (B, K, 3)
    beta_raw = out['beta_raw']         # (B, K, 3)

    pi = F.softmax(pi_logits, dim=1)                               # (B, K)
    alpha, beta = activate_gamma_alpha_beta(
        alpha_raw, beta_raw, eps=eps,
        log_alpha_max=log_alpha_max, log_beta_max=log_beta_max,
        **act_kwargs,
    )
    lam = alpha / beta                                              # (B, K, 3)
    lam_total = lam.sum(dim=-1)                                     # (B, K)

    p_k0 = torch.exp(-lam_total)                                    # (B, K)
    p_ki = (1 - torch.exp(-lam_total)).unsqueeze(-1) * lam / (lam_total.unsqueeze(-1) + eps)
    p_k = torch.cat([p_k0.unsqueeze(-1), p_ki], dim=-1)            # (B, K, 4)

    prob = (pi.unsqueeze(-1) * p_k).sum(dim=1)                     # (B, 4)

    return {
        'prob': prob,
        'logits': torch.log(prob + eps),
        'pred_class': prob.argmax(dim=-1),
    }


def poisson_exact_predict_from_output(out, eps=1e-8):
    """Poisson-Gamma MDN predict with exact Gamma-Poisson marginal.

    Uses softplus activation (gradient bounded, stable) and
    exact P(no mutation) = Π_i (β_i/(β_i+1))^α_i.
    """
    pi_logits = out['pi_logits']       # (B, K)
    alpha_raw = out['alpha_raw']       # (B, K, 3)
    beta_raw = out['beta_raw']         # (B, K, 3)

    pi = F.softmax(pi_logits, dim=1)                               # (B, K)
    alpha = F.softplus(alpha_raw) + eps                             # (B, K, 3)
    beta  = F.softplus(beta_raw) + eps                              # (B, K, 3)

    # P(no mutation) = Π_i (β_i/(β_i+1))^α_i
    log_beta_ratio = -torch.log1p(1.0 / (beta + eps))               # (B, K, 3)
    log_p_k0 = (alpha * log_beta_ratio).sum(dim=-1)                  # (B, K)

    # P(mutation)
    log_p_k_mut = torch.log((-torch.expm1(log_p_k0)).clamp_min(eps))  # (B, K)

    # subtype composition (Gamma mean approx)
    log_lam = torch.log(alpha + eps) - torch.log(beta + eps)          # (B, K, 3)
    log_q = log_lam - torch.logsumexp(log_lam, dim=-1, keepdim=True)  # (B, K, 3)
    log_p_ki = log_p_k_mut.unsqueeze(-1) + log_q                      # (B, K, 3)

    log_p_k = torch.cat([log_p_k0.unsqueeze(-1), log_p_ki], dim=-1)  # (B, K, 4)
    prob = (pi.unsqueeze(-1) * torch.exp(log_p_k)).sum(dim=1)        # (B, 4)

    return {
        'prob': prob,
        'logits': torch.log(prob + eps),
        'pred_class': prob.argmax(dim=-1),
    }


class Network3_ARG_condition_GammaMDN(Network3_ARG_condition):
    """Model 151 variant with Gamma MDN head.

    Key differences from base Network3_ARG_condition:
      - condition_arg is truncated at the 128-dim hidden layer
      - 128-dim hidden is fed into GammaMDNHead (outputs raw values)
      - predict_out['out'] is inferred from pi_logits/alpha_raw/beta_raw
      - Adds predict_out['pi_logits'], predict_out['alpha_raw'], predict_out['beta_raw']

    When gamma_mdn_output_dim=3, uses Poisson-based predict (151_gamma_mdn_poisson).
    When gamma_activation='log', uses log-parameterized activation.
    """

    def __init__(self, *args, K=3, gamma_mdn_output_dim=4, gamma_activation='softplus', **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_mdn_output_dim = gamma_mdn_output_dim
        self.gamma_activation = gamma_activation
        self._log_alpha_max = kwargs.get('config', {}).get('log_alpha_max', 6.0)
        self._log_beta_max = kwargs.get('config', {}).get('log_beta_max', 8.0)
        K = kwargs.get('config', {}).get('K', K)

        # Split condition_arg: keep only the projection to 128-dim
        self.condition_arg_proj = self.condition_arg[:3]  # Linear(68,128)->ReLU->Dropout
        del self.condition_arg

        # Gamma MDN Head (outputs raw values)
        self.mdn_head = GammaMDNHead(
            in_dim=128, K=K, C=gamma_mdn_output_dim,
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
        if self.gamma_mdn_output_dim == 3 and self.gamma_activation == 'log':
            inferred = poisson_gamma_log_predict_from_output(
                predict_out,
                log_alpha_max=self._log_alpha_max,
                log_beta_max=self._log_beta_max,
            )
        elif self.gamma_mdn_output_dim == 3 and self.gamma_activation == 'exact':
            inferred = poisson_exact_predict_from_output(predict_out)
        elif self.gamma_mdn_output_dim == 3:
            inferred = poisson_gamma_mdn_predict_from_output(predict_out)
        else:
            inferred = gamma_mdn_predict_from_output(predict_out)
        predict_out['out'] = inferred['logits']

        if 'local_h1' in local_outs:
            predict_out['local_h1'] = local_outs['local_h1']
            predict_out['local_h2'] = local_outs['local_h2']

        return predict_out, None


# ──────────────────────────────────────────────
# Gamma-Total-Dirichlet MDN
# ──────────────────────────────────────────────

class GammaTotalDirichletMDNHead(nn.Module):
    """Gamma-Total-Dirichlet MDN head.

    Separates total mutation intensity (Gamma, scalar per component) from
    subtype allocation (Dirichlet).

    Args:
        in_dim: Input feature dimension (128)
        K: Number of mixture components (default 3)
        hidden_dim: Shared hidden layer dimension (default 64)
    """
    def __init__(self, in_dim, K=3, hidden_dim=64):
        super().__init__()
        self.K = K

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden_dim, K)              # (B, K)
        self.gamma_alpha = nn.Linear(hidden_dim, K)     # (B, K), scalar rate
        self.gamma_beta = nn.Linear(hidden_dim, K)      # (B, K)
        self.dir_alpha = nn.Linear(hidden_dim, K * 3)   # (B, K, 3)

    def forward(self, x):
        h = self.backbone(x)
        pi_logits = self.pi(h)                                          # [B, K]
        gamma_alpha_raw = self.gamma_alpha(h)                           # [B, K]
        gamma_beta_raw = self.gamma_beta(h)                             # [B, K]
        dir_alpha_raw = self.dir_alpha(h).view(-1, self.K, 3)           # [B, K, 3]
        return pi_logits, gamma_alpha_raw, gamma_beta_raw, dir_alpha_raw


def gamma_total_dirichlet_subtype_predict_from_output(out, eps=1e-8):
    """Infer final predictions from Gamma-Total-Dirichlet MDN model output.

    Returns prob [B,4] and subtype_mix [B,3].
    P(no mutation) = exp(-λ) (Poisson approximation).
    """
    pi_logits = out['pi_logits']                  # (B, K)
    gamma_alpha_raw = out['gamma_alpha_raw']      # (B, K)
    gamma_beta_raw = out['gamma_beta_raw']        # (B, K)
    dir_alpha_raw = out['dir_alpha_raw']          # (B, K, 3)

    pi = F.softmax(pi_logits, dim=1)                                # (B, K)
    gamma_alpha = F.softplus(gamma_alpha_raw) + eps                 # (B, K)
    gamma_beta = F.softplus(gamma_beta_raw) + eps                   # (B, K)
    lam = gamma_alpha / gamma_beta                                   # (B, K), total rate

    dir_alpha = F.softplus(dir_alpha_raw) + eps                     # (B, K, 3)
    subtype_k = dir_alpha / dir_alpha.sum(dim=-1, keepdim=True)     # (B, K, 3)

    p_k0 = torch.exp(-lam)                                           # (B, K)
    p_ki = (1 - torch.exp(-lam)).unsqueeze(-1) * subtype_k           # (B, K, 3)
    p_k = torch.cat([p_k0.unsqueeze(-1), p_ki], dim=-1)             # (B, K, 4)

    prob = (pi.unsqueeze(-1) * p_k).sum(dim=1)                      # (B, 4)
    subtype_mix = (pi.unsqueeze(-1) * subtype_k).sum(dim=1)         # (B, 3)

    return {
        'prob': prob,
        'subtype_mix': subtype_mix,
        'logits': torch.log(prob + eps),
        'pred_class': prob.argmax(dim=-1),
    }


def gamma_total_dirichlet_exact_predict_from_output(out, eps=1e-8):
    """Infer predictions from Gamma-Total-Dirichlet MDN with exact P(0).

    P(no mutation) = (β/(β+1))^α  (exact Gamma-Poisson marginal).
    """
    pi_logits = out['pi_logits']                  # (B, K)
    gamma_alpha_raw = out['gamma_alpha_raw']      # (B, K)
    gamma_beta_raw = out['gamma_beta_raw']        # (B, K)
    dir_alpha_raw = out['dir_alpha_raw']          # (B, K, 3)

    pi = F.softmax(pi_logits, dim=1)                                # (B, K)
    gamma_alpha = F.softplus(gamma_alpha_raw) + eps                 # (B, K)
    gamma_beta = F.softplus(gamma_beta_raw) + eps                   # (B, K)

    # Exact P(no mutation) = (β/(β+1))^α
    log_beta_ratio = -torch.log1p(1.0 / (gamma_beta + eps))         # (B, K)
    log_p_k0 = gamma_alpha * log_beta_ratio                          # (B, K)
    log_p_k_mut = torch.log((-torch.expm1(log_p_k0)).clamp_min(eps)) # (B, K)

    dir_alpha = F.softplus(dir_alpha_raw) + eps                     # (B, K, 3)
    subtype_k = dir_alpha / dir_alpha.sum(dim=-1, keepdim=True)     # (B, K, 3)
    log_p_ki = log_p_k_mut.unsqueeze(-1) + torch.log(subtype_k + eps)  # (B, K, 3)

    log_p_k = torch.cat([log_p_k0.unsqueeze(-1), log_p_ki], dim=-1) # (B, K, 4)
    prob = (pi.unsqueeze(-1) * torch.exp(log_p_k)).sum(dim=1)       # (B, 4)
    subtype_mix = (pi.unsqueeze(-1) * subtype_k).sum(dim=1)         # (B, 3)

    return {
        'prob': prob,
        'subtype_mix': subtype_mix,
        'logits': torch.log(prob + eps),
        'pred_class': prob.argmax(dim=-1),
    }


class Network3_ARG_condition_GammaTotalDirichletMDN(Network3_ARG_condition):
    """Model 151 variant with Gamma-Total-Dirichlet MDN head.

    Gamma models total mutation intensity (scalar per component),
    Dirichlet allocates mutation probability across 3 subtypes.
    """

    def __init__(self, *args, K=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.condition_arg_proj = self.condition_arg[:3]
        del self.condition_arg
        K = kwargs.get('config', {}).get('K', K)

        self.mdn_head = GammaTotalDirichletMDNHead(in_dim=128, K=K)

    def forward(self, local_input, distal_input, arg_feature):
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
        )

        pi_logits, gamma_alpha_raw, gamma_beta_raw, dir_alpha_raw = \
            self.mdn_head(hidden_128)
        predict_out['pi_logits'] = pi_logits             # [B, K]
        predict_out['gamma_alpha_raw'] = gamma_alpha_raw # [B, K]
        predict_out['gamma_beta_raw'] = gamma_beta_raw   # [B, K]
        predict_out['dir_alpha_raw'] = dir_alpha_raw     # [B, K, 3]

        inferred = gamma_total_dirichlet_subtype_predict_from_output(predict_out)
        predict_out['out'] = inferred['logits']

        if 'local_h1' in local_outs:
            predict_out['local_h1'] = local_outs['local_h1']
            predict_out['local_h2'] = local_outs['local_h2']

        return predict_out, None


# ──────────────────────────────────────────────
# Independent-head Gamma MDN (no shared weights across mutation types)
# ──────────────────────────────────────────────

class GammaIndependentHead(nn.Module):
    """Gamma MDN head with per-type independent alpha/beta heads.

    Each mutation type has its own Linear(hidden_dim, K) for alpha
    and beta, so gradient updates for one type never affect another
    type's weights.  This prevents gradient interference between
    high-rate and low-rate mutation types.
    """

    def __init__(self, in_dim, K=3, C_mut=3, hidden_dim=64):
        super().__init__()
        self.K = K
        self.C_mut = C_mut

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden_dim, K)

        for c in range(C_mut):
            setattr(self, f'alpha_{c}', nn.Linear(hidden_dim, K))
            setattr(self, f'beta_{c}', nn.Linear(hidden_dim, K))

    def forward(self, x):
        h = self.backbone(x)
        pi_logits = self.pi(h)                                     # (B, K)

        alphas, betas = [], []
        for c in range(self.C_mut):
            alphas.append(getattr(self, f'alpha_{c}')(h))          # (B, K)
            betas.append(getattr(self, f'beta_{c}')(h))            # (B, K)

        alpha_raw = torch.stack(alphas, dim=-1)                    # (B, K, C)
        beta_raw = torch.stack(betas, dim=-1)                      # (B, K, C)
        return pi_logits, alpha_raw, beta_raw


class Network3_ARG_condition_GammaIndependentMDN(Network3_ARG_condition):
    """Model 151 variant with independent per-type Gamma heads."""

    def __init__(self, *args, K=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.condition_arg_proj = self.condition_arg[:3]
        del self.condition_arg

        self._log_alpha_max = kwargs.get('config', {}).get('log_alpha_max', 6.0)
        self._log_beta_max = kwargs.get('config', {}).get('log_beta_max', 8.0)

        self.mdn_head = GammaIndependentHead(in_dim=128, K=K, C_mut=3)

    def forward(self, local_input, distal_input, arg_feature):
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
        )

        pi_logits, alpha_raw, beta_raw = self.mdn_head(hidden_128)
        predict_out['pi_logits'] = pi_logits
        predict_out['alpha_raw'] = alpha_raw
        predict_out['beta_raw'] = beta_raw

        inferred = poisson_gamma_log_predict_from_output(
            predict_out,
            log_alpha_max=self._log_alpha_max,
            log_beta_max=self._log_beta_max,
        )
        predict_out['out'] = inferred['logits']

        if 'local_h1' in local_outs:
            predict_out['local_h1'] = local_outs['local_h1']
            predict_out['local_h2'] = local_outs['local_h2']

        return predict_out, None
