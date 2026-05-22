"""
Smoke test: verify equivalence between the original for-loop
gamma_mdn_classification_loss and the batched GammaMDNClassificationLoss.

Usage:
    python test_gamma_mdn_loss_equivalence.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Original (user-provided, reference implementation)
# ============================================================

def gamma_mdn_classification_loss_original(pi, alpha, beta, y):
    """
    pi: (B, K)          activated (post-softmax)
    alpha, beta: (B, K, C)  activated (post-softplus, >0)
    y: (B,) long, {0,1,2,3}
    """
    B, K, _ = alpha.shape
    losses = []

    for k in range(K):
        lam = alpha[:, k] / beta[:, k]
        lam = lam / lam.sum(dim=1, keepdim=True)

        log_p = torch.log(lam + 1e-8)
        nll = F.nll_loss(log_p, y, reduction='none')
        losses.append(nll)

    losses = torch.stack(losses, dim=1)           # (B, K)
    log_pi = torch.log(pi + 1e-8)

    return -torch.logsumexp(log_pi - losses, dim=1).mean()


# ============================================================
# Batched (plan implementation)
# ============================================================

class GammaMDNClassificationLoss(nn.Module):
    """Batched Gamma-MDN classification loss.

    Follows the plan: raw inputs -> internal activation -> NLL.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = 'sum'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, y):
        pi_logits, alpha_raw, beta_raw = self._unpack_pred(pred)
        B, K, C = alpha_raw.shape

        # --- Activation ---
        log_pi = F.log_softmax(pi_logits, dim=1)            # (B, K)

        alpha = F.softplus(alpha_raw) + self.eps             # (B, K, C)
        beta = F.softplus(beta_raw) + self.eps               # (B, K, C)

        # --- log p_k(c) via numerically stable log_softmax ---
        log_lam = torch.log(alpha) - torch.log(beta)         # (B, K, C)
        log_p_k = F.log_softmax(log_lam, dim=-1)             # (B, K, C)

        # --- log p(y) via logsumexp ---
        y_idx = y.view(B, 1, 1).expand(B, K, 1)              # (B, K, 1)
        log_p_y = log_p_k.gather(dim=2, index=y_idx).squeeze(2)  # (B, K)

        log_likelihood = torch.logsumexp(log_pi + log_p_y, dim=1)  # (B,)
        nll_loss = -log_likelihood

        # --- reduction ---
        if self.reduction == 'mean':
            nll_loss = nll_loss.mean()
        elif self.reduction == 'sum':
            nll_loss = nll_loss.sum()
        elif self.reduction != 'none':
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return nll_loss

    @staticmethod
    def _unpack_pred(pred):
        if isinstance(pred, dict):
            return pred['pi_logits'], pred['alpha_raw'], pred['beta_raw']
        if isinstance(pred, (tuple, list)):
            if len(pred) != 3:
                raise ValueError("Tuple pred should be (pi_logits, alpha_raw, beta_raw).")
            return pred[0], pred[1], pred[2]
        raise TypeError(
            "pred should be dict or tuple (pi_logits, alpha_raw, beta_raw)."
        )


# ============================================================
# Test helpers
# ============================================================

def generate_random_inputs(B=32, K=3, C=4, seed=None):
    """Generate raw logits/raw params to be consistent with model output."""
    if seed is not None:
        torch.manual_seed(seed)
    pi_logits = torch.randn(B, K)
    alpha_raw = torch.randn(B, K, C) * 0.5
    beta_raw = torch.randn(B, K, C) * 0.5
    y = torch.randint(0, C, (B,))
    return pi_logits, alpha_raw, beta_raw, y


def batched_loss_as_original(pi_logits, alpha_raw, beta_raw, y, eps=1e-8):
    """Compute batched loss using activated inputs,
    to compare against the original for-loop version.
    """
    pi = F.softmax(pi_logits, dim=1)
    alpha = F.softplus(alpha_raw) + eps
    beta = F.softplus(beta_raw) + eps
    return gamma_mdn_classification_loss_original(pi, alpha, beta, y)


# ============================================================
# Tests
# ============================================================

def test_equivalence_default_shapes():
    """Test equivalence with default shapes (B=32, K=3, C=4)."""
    pi_logits, alpha_raw, beta_raw, y = generate_random_inputs(B=32, K=3, C=4, seed=42)

    loss_original = batched_loss_as_original(pi_logits, alpha_raw, beta_raw, y)

    criterion = GammaMDNClassificationLoss(eps=1e-8, reduction='mean')
    loss_batched = criterion((pi_logits, alpha_raw, beta_raw), y)

    assert torch.allclose(loss_original, loss_batched, rtol=1e-5, atol=1e-7), \
        f"Loss mismatch: original={loss_original:.8f}, batched={loss_batched:.8f}"
    print(f"  PASS: default shapes (32,3,4) — original={loss_original:.8f}, batched={loss_batched:.8f}")


def test_equivalence_various_shapes():
    """Test equivalence with various batch sizes, K, and C."""
    cases = [
        (16, 1, 4),
        (8, 2, 4),
        (64, 5, 4),
        (16, 3, 2),
        (32, 4, 6),
        (128, 3, 4),
        (1, 2, 4),
    ]
    for B, K, C in cases:
        pi_logits, alpha_raw, beta_raw, y = generate_random_inputs(B=B, K=K, C=C, seed=99)

        loss_original = batched_loss_as_original(pi_logits, alpha_raw, beta_raw, y)

        criterion = GammaMDNClassificationLoss(eps=1e-8, reduction='mean')
        loss_batched = criterion((pi_logits, alpha_raw, beta_raw), y)

        assert torch.allclose(loss_original, loss_batched, rtol=1e-5, atol=1e-7), \
            f"Loss mismatch for (B={B},K={K},C={C}): original={loss_original:.8f}, batched={loss_batched:.8f}"
        print(f"  PASS: (B={B},K={K},C={C}) — original={loss_original:.8f}, batched={loss_batched:.8f}")


def test_reduction_sum():
    """Test reduction='sum' yields per-batch sum, matching .sum(dim=0) on 'none' output."""
    pi_logits, alpha_raw, beta_raw, y = generate_random_inputs(B=32, K=3, C=4, seed=77)

    criterion_none = GammaMDNClassificationLoss(eps=1e-8, reduction='none')
    criterion_sum = GammaMDNClassificationLoss(eps=1e-8, reduction='sum')

    loss_none = criterion_none((pi_logits, alpha_raw, beta_raw), y)
    loss_sum = criterion_sum((pi_logits, alpha_raw, beta_raw), y)

    assert torch.allclose(loss_none.sum(), loss_sum, rtol=1e-5, atol=1e-7), \
        f"Reduction mismatch: none.sum()={loss_none.sum():.8f}, sum={loss_sum:.8f}"
    print(f"  PASS: reduction='sum' — none.sum()={loss_none.sum():.8f}, sum={loss_sum:.8f}")


def test_reduction_mean():
    """Test reduction='mean' yields per-batch mean, matching .mean() on 'none' output."""
    pi_logits, alpha_raw, beta_raw, y = generate_random_inputs(B=32, K=3, C=4, seed=77)

    criterion_none = GammaMDNClassificationLoss(eps=1e-8, reduction='none')
    criterion_mean = GammaMDNClassificationLoss(eps=1e-8, reduction='mean')

    loss_none = criterion_none((pi_logits, alpha_raw, beta_raw), y)
    loss_mean = criterion_mean((pi_logits, alpha_raw, beta_raw), y)

    assert torch.allclose(loss_none.mean(), loss_mean, rtol=1e-5, atol=1e-7), \
        f"Reduction mismatch: none.mean()={loss_none.mean():.8f}, mean={loss_mean:.8f}"
    print(f"  PASS: reduction='mean' — none.mean()={loss_none.mean():.8f}, mean={loss_mean:.8f}")


def test_dict_pred_format():
    """Test that dict pred format works."""
    pi_logits, alpha_raw, beta_raw, y = generate_random_inputs(B=16, K=2, C=4, seed=1)

    pred_dict = {
        'pi_logits': pi_logits,
        'alpha_raw': alpha_raw,
        'beta_raw': beta_raw,
    }
    criterion = GammaMDNClassificationLoss(eps=1e-8, reduction='mean')
    loss_dict = criterion(pred_dict, y)
    loss_tuple = criterion((pi_logits, alpha_raw, beta_raw), y)

    assert torch.allclose(loss_dict, loss_tuple, rtol=1e-7), \
        f"Dict/tuple mismatch: dict={loss_dict:.8f}, tuple={loss_tuple:.8f}"
    print(f"  PASS: dict pred format — dict={loss_dict:.8f}, tuple={loss_tuple:.8f}")


def test_gradient_flow():
    """Test that gradients flow through all raw parameters."""
    pi_logits = torch.randn(8, 3, requires_grad=True)
    alpha_raw = torch.randn(8, 3, 4, requires_grad=True)
    beta_raw = torch.randn(8, 3, 4, requires_grad=True)
    y = torch.randint(0, 4, (8,))

    criterion = GammaMDNClassificationLoss(eps=1e-8, reduction='mean')
    loss = criterion((pi_logits, alpha_raw, beta_raw), y)
    loss.backward()

    assert pi_logits.grad is not None and pi_logits.grad.abs().sum() > 0, \
        "pi_logits gradient is None or zero"
    assert alpha_raw.grad is not None and alpha_raw.grad.abs().sum() > 0, \
        "alpha_raw gradient is None or zero"
    assert beta_raw.grad is not None and beta_raw.grad.abs().sum() > 0, \
        "beta_raw gradient is None or zero"
    print(f"  PASS: gradient flow — pi_grad_norm={pi_logits.grad.norm():.6f}, "
          f"alpha_grad_norm={alpha_raw.grad.norm():.6f}, beta_grad_norm={beta_raw.grad.norm():.6f}")


# ============================================================
# Run
# ============================================================

if __name__ == '__main__':
    print("Gamma MDN Loss Equivalence Tests\n")

    test_equivalence_default_shapes()
    test_equivalence_various_shapes()
    test_reduction_sum()
    test_reduction_mean()
    test_dict_pred_format()
    test_gradient_flow()

    print(f"\nAll {6} tests passed.")
