# Gamma MDN Poisson Exact 方案

## 核心动机

**精确 Gamma-Poisson 边际**替代泊松近似：

```
泊松近似:  P(非突变) = exp(-Σα/β)              # 仅依赖 α/β 比值
精确边际:  P(非突变) = Π (β/(β+1))^α           # 依赖 α、β 绝对值
```

精确公式让 α 和 β 的绝对值参与 loss，缩小参数空间，有缓解过拟合的潜力。

---

## 实验 1：log-参数化 + exact（已完成）

### 新增代码

**1. `losses.py` — `PoissonExactClassificationLoss`**

```python
class PoissonExactClassificationLoss(nn.Module):
    def forward(self, pred, y):
        ...
        from MuRaL.models.gamma_mdn_model import activate_gamma_alpha_beta
        alpha, beta = activate_gamma_alpha_beta(alpha_raw, beta_raw, ...)  # log-param
        log_beta_ratio = -torch.log1p(1.0 / (beta + self.eps))
        log_p_k0 = (alpha * log_beta_ratio).sum(dim=-1)                 # exact P(no mut)
        ...
```

**2. `gamma_mdn_model.py` — `poisson_exact_predict_from_output`**

```python
def poisson_exact_predict_from_output(out, ...):
    ...
    alpha, beta = activate_gamma_alpha_beta(alpha_raw, beta_raw, ...)   # log-param
    log_beta_ratio = -torch.log1p(1.0 / (beta + eps))
    log_p_k0 = (alpha * log_beta_ratio).sum(dim=-1)
    ...
```

**3. `gamma_mdn_model.py` — forward 分支**

```python
elif self.gamma_mdn_output_dim == 3 and self.gamma_activation == 'exact':
    inferred = poisson_exact_predict_from_output(predict_out, ...)
```

**4. 检测与校验** — `trainingv2.py`、`run_predict.py`

### 结果

首 epoch Batch Var=71515，λ 分布异常（prob3 样本最多但 λ=0.002）。

---

## 实验 2：softplus + exact（下一步）

只改激活函数，exact 公式不变。

### 1. `losses.py` — 激活换为 softplus

```python
class PoissonExactClassificationLoss(nn.Module):
    def forward(self, pred, y):
        pi_logits, alpha_raw, beta_raw = self._unpack_pred(pred)
        B, K, C_mut = alpha_raw.shape
        log_pi = F.log_softmax(pi_logits, dim=1)

        # Softplus 激活（替代 log-参数化）
        alpha = F.softplus(alpha_raw) + self.eps        # (B, K, 3)
        beta  = F.softplus(beta_raw) + self.eps         # (B, K, 3)

        # 以下 exact 公式不变
        log_beta_ratio = -torch.log1p(1.0 / (beta + self.eps))
        log_p_k0 = (alpha * log_beta_ratio).sum(dim=-1)
        log_p_k_mut = torch.log((-torch.expm1(log_p_k0)).clamp_min(self.eps))
        log_lam = torch.log(alpha + self.eps) - torch.log(beta + self.eps)
        log_q = log_lam - torch.logsumexp(log_lam, dim=-1, keepdim=True)
        log_p_ki = log_p_k_mut.unsqueeze(-1) + log_q
        log_p_k = torch.cat([log_p_k0.unsqueeze(-1), log_p_ki], dim=-1)
        y_idx = y.view(B, 1, 1).expand(B, K, 1)
        log_p_y = log_p_k.gather(dim=2, index=y_idx).squeeze(2)
        log_likelihood = torch.logsumexp(log_pi + log_p_y, dim=1)
        nll_loss = -log_likelihood
        ...
```

### 2. `gamma_mdn_model.py` — predict 函数改为 softplus，保留 exact 公式

```python
# 改动前（log-参数化）
alpha, beta = activate_gamma_alpha_beta(alpha_raw, beta_raw, ...)

# 改动后（softplus）
alpha = F.softplus(alpha_raw) + eps
beta  = F.softplus(beta_raw) + eps
# exact 公式不变
log_beta_ratio = -torch.log1p(1.0 / (beta + eps))
log_p_k0 = (alpha * log_beta_ratio).sum(dim=-1)
...
```

forward 分支保留，`gamma_activation == 'exact'` 继续触发 `poisson_exact_predict_from_output`。

### 3. `model_config.py` — 清理配置

删除 `'gamma_activation': 'exact'` 等 log 相关参数。

### 验证标准

- 首 epoch Batch Var < 1000
- λ 分布与数据中样本比例一致
- prob1/prob2/prob3 entropy calibration 全部正值

---

## 实验 3：Gamma-Total-Dirichlet + exact

在 `151_gamma_total_dirichlet_mdn_k1` 基础上，将其 P(非突变) 从泊松近似改为精确 Gamma-Poisson 边际。

### 数学差异

```
# 实验 2（exact 泊松）：3 个独立 Gamma，P(0) = Π(β_i/(β_i+1))^α_i
# 实验 3（exact 总强度）：1 个 Gamma（总强度）+ Dirichlet（方向）
#           P(0) = (β_γ / (β_γ + 1))^α_γ
#           P(type i|mut) = dir_α_i / Σ dir_α_j
```

Gamma-Total-Dirichlet 中 α_γ、β_γ 是**标量**（每分量一个总强度），不是向量。P(非突变) 只依赖总强度，方向分配由 Dirichlet 独立处理。

### 新增/修改

- **Loss**: `GammaTotalDirichletExactLoss` — 保留三段式（L_4class + w_mut·BCE + w_type·CE），但 L_4class 中 P(非突变) = (β/(β+1))^α
- **Predict**: 在 `gamma_total_dirichlet_subtype_predict_from_output` 中替换 P(0) 公式
- **Model config**: `151_gamma_total_dirichlet_mdn_exact` 分支，K=1

### 改动范围

| # | 文件 | 改动 |
|---|------|------|
| 1 | `losses.py` | 新增 `GammaTotalDirichletExactLoss`，注册 + AdaptiveLossStrategy2 |
| 2 | `gamma_mdn_model.py` | 新增 `gamma_total_dirichlet_exact_predict_from_output` |
| 3 | `model_config.py` | 新增 `151_gamma_total_dirichlet_mdn_exact` 分支 |
| 4 | `trainingv2.py` | 检测 + 校验 |
| 5 | `run_predict.py` | 检测 + 校验 + predictor + 未激活文件 |

observer.py、predict.py 无需修改。`GammaTotalDirichletRecoder` 的 key 名与原 Total Dirichlet 一致。`GammaLambdaRecoder` 也可复用（γ/β 仍是 gamma_alpha_raw/gamma_beta_raw）。
