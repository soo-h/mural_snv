"""
Comprehensive tests for Poisson-Gamma MDN extension.

Tests:
  1. PoissonGammaMDNClassificationLoss forward & reduction
  2. poisson_gamma_mdn_predict_from_output shapes & monotonicity
  3. Loss-predict consistency (prob from loss == prob from predict)
  4. Regression: existing GammaMDNClassificationLoss unchanged
  5. Edge case: gamma_mdn_output_dim=3 vs dim=4 in model

Usage:
    cd /public5/home/songhui/git_repo/mural_snv
    python Develop/DirichletMDN/test_gamma_mdn_poisson.py
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, '/public5/home/songhui/git_repo/mural_snv')

from MuRaL.models.losses import (
    PoissonGammaMDNClassificationLoss,
    GammaMDNClassificationLoss,
)
from MuRaL.models.gamma_mdn_model import (
    poisson_gamma_mdn_predict_from_output,
    gamma_mdn_predict_from_output,
    GammaMDNHead,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = 1e-6
n_pass = 0
n_fail = 0


def check(cond, msg):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f'  PASS: {msg}')
    else:
        n_fail += 1
        print(f'  FAIL: {msg}')


# ============================================================
# Helpers
# ============================================================

def make_pred(B=4, K=3, C=3):
    """Create a random Poisson-Gamma MDN pred dict with C=3."""
    return {
        'pi_logits': torch.randn(B, K),
        'alpha_raw': torch.randn(B, K, C) * 0.5,
        'beta_raw': torch.randn(B, K, C) * 0.5,
    }


# ============================================================
# 1. PoissonGammaMDNClassificationLoss forward
# ============================================================
print('=== Test 1: PoissonGammaMDNClassificationLoss forward ===')

loss_fn = PoissonGammaMDNClassificationLoss(reduction='sum')

# 1a. Basic shapes
pred = make_pred(B=4, K=3, C=3)
y = torch.randint(0, 4, (4,))
loss = loss_fn(pred, y)
check(loss.shape == () and loss.numel() == 1, f'loss shape (), got {loss.shape}')
check(loss.item() > 0, f'loss positive ({loss.item():.4f})')

# 1b. Reduction='mean'
loss_mean = PoissonGammaMDNClassificationLoss(reduction='mean')(pred, y)
check(abs(loss_mean.item() - loss.item() / 4) < 1e-5,
      f'mean loss matches sum/4 ({loss_mean.item():.6f} vs {loss.item()/4:.6f})')

# 1c. Different B sizes
for B in [1, 2, 8]:
    p = make_pred(B=B, K=3, C=3)
    lbl = torch.randint(0, 4, (B,))
    l = loss_fn(p, lbl)
    check(l.item() > 0, f'B={B} loss positive ({l.item():.4f})')

# 1d. Known deterministic case: if all predictions are the same
torch.manual_seed(42)
pred_fixed = {
    'pi_logits': torch.zeros(2, 3),  # uniform pi
    'alpha_raw': torch.ones(2, 3, 3) * 0.5,
    'beta_raw': torch.ones(2, 3, 3) * 0.5,
}
y_fixed = torch.tensor([0, 1])
l_fixed = loss_fn(pred_fixed, y_fixed)
check(torch.isfinite(l_fixed), f'deterministic case loss finite ({l_fixed.item():.4f})')


# ============================================================
# 2. poisson_gamma_mdn_predict_from_output
# ============================================================
print('\n=== Test 2: poisson_gamma_mdn_predict_from_output ===')

pred = make_pred(B=4, K=3, C=3)
result = poisson_gamma_mdn_predict_from_output(pred)

# 2a. Output keys
check('prob' in result, 'has prob')
check('logits' in result, 'has logits')
check('pred_class' in result, 'has pred_class')

# 2b. Shapes
check(result['prob'].shape == (4, 4), f'prob shape (4,4), got {result["prob"].shape}')
check(result['logits'].shape == (4, 4), f'logits shape (4,4), got {result["logits"].shape}')
check(result['pred_class'].shape == (4,), f'pred_class shape (4,), got {result["pred_class"].shape}')

# 2c. Probabilities sum to 1
prob_sum = result['prob'].sum(dim=1)
check(torch.allclose(prob_sum, torch.ones(4), atol=1e-5),
      f'prob sums to 1 ({prob_sum.tolist()})')

# 2d. logits consistent with prob
check((result['logits'] <= 0).all(), 'logits <= 0')
log_prob = torch.log(result['prob'] + eps)
max_logit_diff = (result['logits'] - log_prob).abs().max().item()
check(torch.allclose(result['logits'], log_prob, atol=1e-4),
      f'logits ≈ log(prob) (max diff: {max_logit_diff:.6f})')

# 2e. All values positive
check(torch.all(result['prob'] >= 0) and torch.all(result['prob'] <= 1),
      'prob in [0, 1]')

# 2f. pred_class is argmax
check(torch.equal(result['pred_class'], result['prob'].argmax(dim=-1)),
      'pred_class == argmax')


# ============================================================
# 3. Loss-predict consistency
# ============================================================
print('\n=== Test 3: Loss-predict consistency ===')

# Verify that loss computes the same probabilities as predict.
# The NLL of the true class equals -log(prob[y]) from the predict function.
for B in [1, 3, 8]:
    pred = make_pred(B=B, K=3, C=3)
    y = torch.randint(0, 4, (B,))

    # Reference: prob from predict
    ref = poisson_gamma_mdn_predict_from_output(pred)
    ref_nll = -torch.log(ref['prob'][torch.arange(B), y] + eps).sum()

    # Loss output (reduction='sum')
    l = PoissonGammaMDNClassificationLoss(reduction='sum')(pred, y)
    check(torch.allclose(l, ref_nll, atol=1e-4),
          f'B={B} loss matches -sum(log prob(y)) ({l.item():.4f} vs {ref_nll.item():.4f})')


# ============================================================
# 4. Regression: existing GammaMDNClassificationLoss
# ============================================================
print('\n=== Test 4: Regression - GammaMDNClassificationLoss unchanged ===')

pred_gmdn = make_pred(B=4, K=3, C=4)
y = torch.randint(0, 4, (4,))
loss_gmdn = GammaMDNClassificationLoss(reduction='sum')(pred_gmdn, y)
check(loss_gmdn.item() > 0, f'GammaMDN loss positive ({loss_gmdn.item():.4f})')

# Compare with reference predict
ref_gmdn = gamma_mdn_predict_from_output(pred_gmdn)
ref_nll_gmdn = -torch.log(ref_gmdn['prob'][torch.arange(4), y] + eps).sum()
check(torch.allclose(loss_gmdn, ref_nll_gmdn, atol=1e-4),
      f'GammaMDN loss matches -sum(log prob(y)) ({loss_gmdn.item():.4f} vs {ref_nll_gmdn.item():.4f})')


# ============================================================
# 5. Edge cases
# ============================================================
print('\n=== Test 5: Edge cases ===')

# 5a. Very small lambda (near-zero mutation rates)
pred_small = {
    'pi_logits': torch.randn(2, 3),
    'alpha_raw': torch.full((2, 3, 3), -10.0),  # alpha → 0 after softplus
    'beta_raw': torch.full((2, 3, 3), 1.0),
}
y_small = torch.tensor([0, 1])
loss_small = PoissonGammaMDNClassificationLoss(reduction='sum')(pred_small, y_small)
check(torch.isfinite(loss_small), f'small lambda loss finite ({loss_small.item():.4f})')

# 5b. All pi weight on one component
pred_onecomp = {
    'pi_logits': torch.tensor([[10.0, 0.0, 0.0]]),  # pi ≈ [1, 0, 0]
    'alpha_raw': torch.ones(1, 3, 3) * 0.5,
    'beta_raw': torch.ones(1, 3, 3) * 0.5,
}
y_onecomp = torch.tensor([1])
loss_onecomp = PoissonGammaMDNClassificationLoss(reduction='sum')(pred_onecomp, y_onecomp)
result_onecomp = poisson_gamma_mdn_predict_from_output(pred_onecomp)
check(torch.isfinite(loss_onecomp), 'single component loss finite')
check(torch.allclose(result_onecomp['prob'].sum(dim=1), torch.ones(1), atol=1e-5),
      'single component prob sums to 1')

# 5c. C=3 GammaMDNHead forward
head = GammaMDNHead(in_dim=128, K=3, C=3)
x = torch.randn(4, 128)
pi_logits, alpha_raw, beta_raw = head(x)
check(pi_logits.shape == (4, 3), f'pi_logits shape (4,3), got {pi_logits.shape}')
check(alpha_raw.shape == (4, 3, 3), f'alpha_raw shape (4,3,3), got {alpha_raw.shape}')
check(beta_raw.shape == (4, 3, 3), f'beta_raw shape (4,3,3), got {beta_raw.shape}')

# 5d. C=4 GammaMDNHead forward (backward compat)
head_c4 = GammaMDNHead(in_dim=128, K=3, C=4)
pi_logits_c4, alpha_raw_c4, beta_raw_c4 = head_c4(x)
check(alpha_raw_c4.shape == (4, 3, 4), f'C=4 alpha_raw shape (4,3,4), got {alpha_raw_c4.shape}')


# ============================================================
# 6. Poisson: monotonicity
# ============================================================
print('\n=== Test 6: Monotonicity ===')

# With fixed alpha/beta, larger lambda → smaller p(0)
pred_low = {
    'pi_logits': torch.zeros(1, 3),
    'alpha_raw': torch.full((1, 3, 3), 0.01),
    'beta_raw': torch.full((1, 3, 3), 1.0),
}
pred_high = {
    'pi_logits': torch.zeros(1, 3),
    'alpha_raw': torch.full((1, 3, 3), 2.0),
    'beta_raw': torch.full((1, 3, 3), 1.0),
}
p_low = poisson_gamma_mdn_predict_from_output(pred_low)['prob']
p_high = poisson_gamma_mdn_predict_from_output(pred_high)['prob']
check(p_low[0, 0].item() > p_high[0, 0].item(),
      f'lower lambda → higher p(0) ({p_low[0,0]:.4f} > {p_high[0,0]:.4f})')
check(p_low[0, 1:].sum().item() < p_high[0, 1:].sum().item(),
      f'higher lambda → higher p(mut) ({p_low[0,1:].sum():.4f} < {p_high[0,1:].sum():.4f})')


# ============================================================
# Summary
# ============================================================
print(f'\n{"="*50}')
print(f'Results: {n_pass} passed, {n_fail} failed')
if n_fail > 0:
    print('SOME TESTS FAILED!')
    sys.exit(1)
else:
    print('All tests passed.')
