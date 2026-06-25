# Gamma-Total-Dirichlet (λ,α) 参数化方案

## 动机

当前 `151_gamma_total_dirichlet_mdn_exact` 使用 (α,β) 参数化：lambda = α/β，P(非突变) = (β/(β+1))^α。

梯度路径：
```
∂L/∂β → ∂β/∂beta_raw (via softplus) → ∂P/∂β = α/β² · (β/(β+1))^(α-1) · 1/(β+1)²
∂L/∂α → ∂α/∂alpha_raw (via softplus) → ∂P/∂α = log(β/(β+1)) · (β/(β+1))^α
```

β 同时参与 ratio（α/β）和 individual 值（log(β/(β+1))），两个任务可能冲突。

改为 (λ,α) 参数化：λ 直接是 mean，α 是 concentration，β = α/λ 为派生量。

梯度路径：
```
∂L/∂λ → 直接 ∂P/∂λ
∂L/∂α → 直接 ∂P/∂α
```

梯度**直接**作用于 λ 和 α，去除了 β 的中间路径。

## 数学对比

```
(α,β) 参数化:        (λ,α) 参数化
λ = α/β              λ = λ（直接）
α = α                α = α
β = β                β = α/λ（派生）

P(0) = (β/(β+1))^α   P(0) = (α/(α+λ))^α

     λ ≥ 0, α > 0          λ ≥ 0, α > 0
```

## 新增组件

| 组件 | 说明 |
|------|------|
| Head | `GammaLambdaAlphaHead` — 输出 lambda_raw [B,K]、alpha_raw [B,K] |
| Model | 复用 `Network3_ARG_condition_GammaTotalDirichletMDN` — 只替换 head + forward |
| Predict | `gamma_lambda_alpha_predict_from_output` — 直接使用 λ |
| Loss | `GammaLambdaAlphaExactLoss` — 与 exact 一致但 λ 直接参与 |
| Config | `151_gamma_total_dirichlet_mdn_exact_lambda`（或更短 `_lambda`）|

### Head 改动

```python
# 当前 (α,β)
self.gamma_alpha = nn.Linear(hidden_dim, K)
self.gamma_beta  = nn.Linear(hidden_dim, K)
# forward → gamma_alpha_raw [B,K], gamma_beta_raw [B,K]

# 新 (λ,α)
self.lambda_mu = nn.Linear(hidden_dim, K)
self.alpha_conc = nn.Linear(hidden_dim, K)
# forward → lambda_raw [B,K], alpha_raw [B,K]
```

### Predict 改动

```python
# 当前 (α,β): P(0) = (β/(β+1))^α
gamma_alpha = softplus(gamma_alpha_raw)
gamma_beta  = softplus(gamma_beta_raw)
lam = gamma_alpha / gamma_beta
log_p_k0 = gamma_alpha * log(gamma_beta/(gamma_beta+1))

# 新 (λ,α): P(0) = (α/(α+λ))^α
lam = softplus(lambda_raw)
alpha_conc = softplus(alpha_raw)
log_p_k0 = -alpha_conc * torch.log1p(lam / (alpha_conc + eps))   # 数值稳定形式
```

### Loss 改动

和 predict 一致的 P(0) 公式，梯度直接作用于 λ。

## 修改清单

| # | 文件 | 改动 |
|---|------|------|
| 1 | `gamma_mdn_model.py` | 新增 `GammaLambdaAlphaHead`、`gamma_lambda_alpha_predict_from_output`、`Network3_ARG_condition_GammaLambdaAlphaMDN` |
| 2 | `losses.py` | 新增 `GammaLambdaAlphaExactLoss`，注册 + AdaptiveLossStrategy2 |
| 3 | `model_config.py` | 新增 `151_gamma_total_dirichlet_mdn_exact_lambda` |
| 4 | `trainingv2.py` | 检测 + 校验 |
| 5 | `run_predict.py` | 检测 + 校验 + predictor + 未激活文件 |

observer.py / predict.py 无需修改。
