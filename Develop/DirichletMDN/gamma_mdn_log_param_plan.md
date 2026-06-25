# Gamma MDN Log-Parameter 拓展方案

## 概述

将 Poisson-Gamma MDN 的激活方式从 softplus 改为 **log-空间 clamp + exp 激活**。key 名（`alpha_raw`/`beta_raw`）不变，但语义变为 log-参数。

| | 现有 softplus | 新 log-参数 |
|--|-------------|------------|
| 激活 | `softplus(raw) + eps` | `alpha_min + exp(clamp(log_alpha, lo, hi))` |
| alpha 下限 | 仅 eps (~1e-8) | `alpha_min=0.05` 显式保证 |
| beta 下限 | 仅 eps | `beta_min=1e-3` 显式保证 |
| clamp 范围 | 无 | 有界 log 空间 |
| Head 结构 | 不变 | 不变（同为 `alpha_raw`/`beta_raw` Linear 头） |
| model_no | `151_gamma_mdn_poisson` | `151_gamma_mdn_poisson_log` |

**动机**: softplus 对极端值处理不够，log 空间 clamp 提供更精确的数值控制。

---

## 1. 激活函数

### 1.1 bounded_exp

`torch.exp(x)` 的梯度也是 `exp(x)`。当 `x > 0` 时梯度随 x 指数增长（`x=3 → grad=20`，`x=6 → grad=403`），
使共享 backbone 的优化被高 λ 类型（如 prob3）完全支配，其他突变方向学不动（实验验证：prob3 entropy calibration corr = -0.98）。

用分段函数 `bounded_exp` 替代：`x < 0` 时保持 `exp(x)`（保留小值信息），`x ≥ 0` 时切换为 `x+1`（梯度恒为 1）。

```python
def _bounded_exp(x):
    """exp(x) for x < 0 (preserves small-value ordering),
       x+1  for x >= 0 (gradient bounded at 1).

    torch.where evaluates both branches, so exp(x) is clamped to ≤0
    to prevent overflow (x ≥ 88 → exp overflow in float32)."""
    return torch.where(x < 0, torch.exp(x.clamp(max=0.)), x + 1.0)
```

`x=0` 处两侧值均为 1，导数均为 1，连续可导。

### 1.2 activate_gamma_alpha_beta

```python
def activate_gamma_alpha_beta(
    alpha_raw,
    beta_raw,
    *,
    alpha_min=0.05,
    beta_min=1e-3,
    log_alpha_min=-10.0,   # 仅下界 clamp，防止 log(0) 问题
    log_beta_min=-10.0,
    eps=1e-8,
):
    log_alpha = alpha_raw.clamp(min=log_alpha_min)
    log_beta  = beta_raw.clamp(min=log_beta_min)

    alpha = alpha_min + _bounded_exp(log_alpha)
    beta  = beta_min  + _bounded_exp(log_beta)

    alpha = torch.clamp(alpha, min=eps)
    beta  = torch.clamp(beta, min=eps)

    return alpha, beta
```

**与现有 softplus 的对比**:
```python
# 现有 (gamma_mdn_predict_from_output, PoissonGammaMDNClassificationLoss)
alpha = F.softplus(alpha_raw) + eps
beta  = F.softplus(beta_raw) + eps

# 新
alpha, beta = activate_gamma_alpha_beta(alpha_raw, beta_raw)
```

---

## 2. 实现策略

**核心思路**：Head 结构完全不变（同 `GammaMDNHead`），仅替换激活函数。通过新 model_no 和 loss 类区分。

| 层级 | 改动 | 说明 |
|------|------|------|
| Head | 无 | 复用 `GammaMDNHead`（C=3），key 名不变 |
| Model | 轻微 | 复用 `Network3_ARG_condition_GammaMDN`，forward 中通过 mode 选择 predict 函数 |
| Loss | 新增 | `PoissonGammaLogClassificationLoss`，使用 `activate_gamma_alpha_beta` |
| Predict | 新增 | `poisson_gamma_log_predict_from_output` |

---

## 3. 详细设计

### 3.1 gamma_mdn_model.py

**新增激活函数**: `activate_gamma_alpha_beta`

**新增 predict**:
```python
def poisson_gamma_log_predict_from_output(out, eps=1e-8, **act_kwargs):
    pi_logits = out['pi_logits']       # (B, K)
    alpha_raw = out['alpha_raw']       # (B, K, 3)
    beta_raw = out['beta_raw']         # (B, K, 3)

    pi = F.softmax(pi_logits, dim=1)
    alpha, beta = activate_gamma_alpha_beta(alpha_raw, beta_raw, eps=eps, **act_kwargs)
    lam = alpha / beta                                            # (B, K, 3)
    lam_total = lam.sum(dim=-1)                                   # (B, K)

    p_k0 = torch.exp(-lam_total)
    p_ki = (1 - torch.exp(-lam_total)).unsqueeze(-1) * lam / (lam_total.unsqueeze(-1) + eps)
    p_k = torch.cat([p_k0.unsqueeze(-1), p_ki], dim=-1)          # (B, K, 4)

    prob = (pi.unsqueeze(-1) * p_k).sum(dim=1)                   # (B, 4)

    return {
        'prob': prob,
        'logits': torch.log(prob + eps),
        'pred_class': prob.argmax(dim=-1),
    }
```

### 3.2 losses.py

**新 loss 类**:
```python
class PoissonGammaLogClassificationLoss(nn.Module):
    def __init__(self, eps=1e-8, reduction='sum',
                 alpha_min=0.05, beta_min=1e-3,
                 log_alpha_min=-10.0, log_beta_min=-10.0):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.log_alpha_min = log_alpha_min
        self.log_beta_min = log_beta_min

    def forward(self, pred, y):
        pi_logits, alpha_raw, beta_raw = self._unpack_pred(pred)
        B, K, C_mut = alpha_raw.shape  # C_mut = 3

        log_pi = F.log_softmax(pi_logits, dim=1)

        alpha, beta = activate_gamma_alpha_beta(
            alpha_raw, beta_raw, eps=self.eps,
            alpha_min=self.alpha_min, beta_min=self.beta_min,
            log_alpha_min=self.log_alpha_min,
            log_beta_min=self.log_beta_min,
        )

        lam = alpha / beta
        lam_total = lam.sum(dim=-1)

        log_p_k0 = -lam_total
        log_1mexp = torch.log(-torch.expm1(-lam_total) + self.eps)
        log_p_ki = (
            log_1mexp.unsqueeze(-1)
            + torch.log(lam + self.eps)
            - torch.log(lam_total.unsqueeze(-1) + self.eps)
        )
        log_p_k = torch.cat([log_p_k0.unsqueeze(-1), log_p_ki], dim=-1)

        y_idx = y.view(B, 1, 1).expand(B, K, 1)
        log_p_y = log_p_k.gather(dim=2, index=y_idx).squeeze(2)
        log_likelihood = torch.logsumexp(log_pi + log_p_y, dim=1)
        nll_loss = -log_likelihood

        if self.reduction == 'mean': nll_loss = nll_loss.mean()
        elif self.reduction == 'sum': nll_loss = nll_loss.sum()
        return nll_loss

    @staticmethod
    def _unpack_pred(pred):
        if isinstance(pred, dict):
            return pred['pi_logits'], pred['alpha_raw'], pred['beta_raw']
        raise TypeError("pred should be dict with pi_logits, alpha_raw, beta_raw")
```

**LossFactory 注册**:
```python
elif loss_name == 'PoissonGammaLog':
    return PoissonGammaLogClassificationLoss(reduction='sum')
```

**AdaptiveLossStrategy2**:
```python
is_poisson_gamma_log = isinstance(criterion, PoissonGammaLogClassificationLoss)
...
if is_dir_mdn or is_gamma_mdn or is_poisson_gamma_mdn or is_gamma_total_dirichlet or is_poisson_gamma_log:
    loss = criterion(preds, y)  # full preds dict
```

### 3.3 model_config.py

```python
elif model_no == '151_gamma_mdn_poisson_log':
    from MuRaL.models.gamma_mdn_model import Network3_ARG_condition_GammaMDN
    model_config = { ... }  # 与 151_gamma_mdn_poisson 相同
    model_specify_config = {
        'fused_type': 'logit',
        'n_arg_features': 23,
        'arg_hidden_dim': 128, 'arg_out_dim': 64,
        'arg_dropout': [0.2, 0.1, 0.1],
        'K': 3,
        'gamma_mdn_output_dim': 3,
        'gamma_activation': 'log',   # 标识 log 参数化
    }
    model = Network3_ARG_condition_GammaMDN(...
        gamma_mdn_output_dim=model_config.get('gamma_mdn_output_dim', 4),
        gamma_activation=model_config.get('gamma_activation', 'softplus'))
```

### 3.4 观察与预测侧

**判断方式**: 用 `isinstance(criterion, PoissonGammaLogClassificationLoss)` 或 model_no 字符串 `'_poisson_log'`。

**Recoder**: `GammaMDNRecoder` 完全复用 — key 名（`alpha_raw`/`beta_raw`/`pi_logits`）不变。

**预测文件命名**: `*_gamma_mdn_log_unactivated.tsv.gz`

---

## 4. 修改清单

| # | 文件 | 修改内容 |
|---|------|----------|
| 1 | `gamma_mdn_model.py` | 新增 `activate_gamma_alpha_beta`、`poisson_gamma_log_predict_from_output`；模型类新增 `gamma_activation` 参数，forward 中根据 mode 选择 predict |
| 2 | `losses.py` | 新增 `PoissonGammaLogClassificationLoss`，注册 + AdaptiveLossStrategy2 |
| 3 | `model_config.py` | 新增 `151_gamma_mdn_poisson_log` 分支 |
| 4 | `trainingv2.py` | 检测 + 校验 |
| 5 | `run_predict.py` | 检测 + 校验 + predictor + 未激活文件 |

**无需修改**: `observer.py`、`predict.py`（`GammaMDNRecoder` 完全兼容）

---

## 5. 设计决策（Grill 确认）

| # | 决策 | 结论 |
|---|------|------|
| Q1 | model_no | `151_gamma_mdn_poisson_log`（poisson 变体的 log-参数化版本） |
| Q2 | forward mode | `gamma_activation` 参数（`'softplus'` 为默认，向后兼容） |
| Q3 | clamp 默认值 | `model_specify_config` 可覆盖，loss 类保留默认值 |
| Q4 | loss 识别 | `PoissonGammaLog` loss_name，`PoissonGammaLogClassificationLoss` 独立类 |

---

## 6. Validation λ 统计输出（Training-only）

### 动机

Validation 时当前输出 pi_entropy（混合权重锐度），但缺少直接的突变强度 λ（= α/β）统计。λ 对理解模型的生物学预测更有价值。

### 实现

**observer.py — 新增 `GammaLambdaRecoder`**

```python
class GammaLambdaRecoder(Observer):
    """Collect mixture-weighted mutation intensity λ during validation.

    λ_{k,i} = α_{k,i} / β_{k,i}, then mixture-weighted to (B, C).
    """

    def __init__(self, gamma_activation='softplus'):
        super().__init__()
        self.lam = None
        self.gamma_activation = gamma_activation

    def reset(self):
        self.lam = None

    def recode(self, preds):
        predict_out, _ = preds if isinstance(preds, tuple) else (preds, None)
        if 'alpha_raw' not in predict_out:
            return
        alpha_raw = predict_out['alpha_raw'].detach()
        beta_raw = predict_out['beta_raw'].detach()

        if self.gamma_activation == 'log':
            from MuRaL.models.gamma_mdn_model import activate_gamma_alpha_beta
            alpha, beta = activate_gamma_alpha_beta(alpha_raw, beta_raw)
        else:
            alpha = F.softplus(alpha_raw) + 1e-8
            beta = F.softplus(beta_raw) + 1e-8

        lam = (alpha / beta).cpu()                               # (B, K, C)
        pi = F.softmax(predict_out['pi_logits'].detach(), dim=1) # (B, K)
        lam_mix = (pi.unsqueeze(-1) * lam).sum(dim=1)            # (B, C)

        self.lam = lam_mix if self.lam is None else torch.cat([self.lam, lam_mix], dim=0)

    def output(self):
        return self.lam

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            self.recode(kwargs['valid_preds'])
```

**trainingv2.py — 注册、提取、打印**

```python
# 创建（与 pi_entropy_recoder 并列）
lambda_recoder = GammaLambdaRecoder(
    gamma_activation='log' if is_poisson_gamma_log else 'softplus'
) if is_any_gamma_mdn else None

# 在 valid_step 前注册
if lambda_recoder:
    trainer.register_observer(lambda_recoder)

# valid_step 后提取 + 打印
if lambda_recoder:
    lam = lambda_recoder.output()
    lam_np = to_np(lam)
    for i in range(3):
        print(f"prob{i+1} lambda: mean={lam_np[:,i].mean():.6f} "
              f"max={lam_np[:,i].max():.6f} min={lam_np[:,i].min():.6f} "
              f"std={lam_np[:,i].std():.6f}")
    lambda_recoder.reset()
    trainer.remove_observer(lambda_recoder)
```

### 修改清单

| # | 文件 | 改动 |
|---|------|------|
| 1 | `MuRaL/evaluation/observer.py` | 新增 `GammaLambdaRecoder` |
| 2 | `MuRaL/scripts/trainingv2.py` | recoder 创建、注册、提取、打印统计 |

**predict.py / run_predict.py 无需修改**（training-only 功能）。
