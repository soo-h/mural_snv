# Gamma MDN 拓展方案

## 概述

在 **model 151** (`Network3_ARG_condition`) 基础上引入 **Gamma Mixture Density Network (MDN)** 头，通过继承方式实现，与 Dirichlet MDN 的拓展模式一致。

**Gamma MDN vs Dirichlet MDN 核心区别**：

| 特性 | Dirichlet MDN | Gamma MDN |
|------|---------------|-----------|
| 分布参数 | `alpha` (浓度) | `alpha` (shape) + `beta` (rate) |
| 类概率推导 | Dirichlet 期望: `alpha_c / sum(alpha)` | Gamma 期望归一化: `(alpha/beta) / sum(alpha/beta)` |
| 不确定性 | 天然 evidence (alpha sum) | pi 的熵 `-Σ pi·log pi` |
| 输出头 | pi + alpha (2 个 Linear) | pi + alpha + beta (3 个 Linear) |

**关键设计原则**：model forward 始终输出原始值（`pi_logits`, `alpha_raw`, `beta_raw`），不做任何激活。激活分别由 loss 和 `predict_from_output` 各自负责。

---

## 1. GammaMDNHead

输出原始值，不激活。激活由 loss 和 `predict_from_output` 各自独立完成。

### 架构

```text
in_dim (128)
    ↓
Shared Backbone: Linear(128, 64) → ReLU
    ↓
  ├── pi_linear(64, K)           → pi_logits     [batch, K]         (raw)
  ├── alpha_linear(64, K*C)      → alpha_raw     [batch, K, C]      (raw)
  └── beta_linear(64, K*C)       → beta_raw      [batch, K, C]      (raw)
```

### 实现

```python
class GammaMDNHead(nn.Module):
    """Gamma Mixture Density Network head.

    Architecture: in_dim → shared MLP → pi_head / alpha_head / beta_head.
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
        self.pi = nn.Linear(hidden_dim, K)            # → pi_logits (raw)
        self.alpha = nn.Linear(hidden_dim, K * C)     # → alpha_raw (raw)
        self.beta = nn.Linear(hidden_dim, K * C)      # → beta_raw (raw)

    def forward(self, x):
        h = self.backbone(x)                                        # [batch, hidden_dim]
        pi_logits = self.pi(h)                                      # [batch, K]
        alpha_raw = self.alpha(h).view(-1, self.K, self.C)          # [batch, K, C]
        beta_raw = self.beta(h).view(-1, self.K, self.C)            # [batch, K, C]
        return pi_logits, alpha_raw, beta_raw
```

---

## 2. in_dim 连接位置

与 Dirichlet MDN 完全相同。复用 `condition_arg` 的 128-dim 隐藏层。

### 拆分方式

```python
# 原 Sequential
self.condition_arg = nn.Sequential(
    nn.Linear(n_class + arg_out_dim, 128),   # [0]
    nn.ReLU(),                                # [1]
    nn.Dropout(0.1),                          # [2]
    nn.Linear(128, n_class),                  # [3]  → 不再使用
)

# 拆分为
self.condition_arg_proj = nn.Sequential(     # [0, 1, 2]
    nn.Linear(n_class + arg_out_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
)  # → [batch, 128] 作为 GammaMDNHead 的输入
```

---

## 3. 模型类设计

### 3.1 继承结构

```text
Network3_ARG_condition (model 151)
  ├── Network3_ARG_condition_NB         (151_nb)           — NB v1
  ├── Network3_ARG_condition_NBv2       (151_nb_v2)        — NB v2
  ├── Network3_ARG_condition_NBv3       (151_nb_v3)        — NB v3
  ├── Network3_ARG_condition_DirMDN     (151_dir_mdn)      — Dirichlet MDN
  └── Network3_ARG_condition_GammaMDN   (151_gamma_mdn)    — Gamma MDN  <-- 新增
```

### 3.2 关键实现

```python
class Network3_ARG_condition_GammaMDN(Network3_ARG_condition):
    """Model 151 variant with Gamma MDN head.

    与普通 151 的关键区别:
      - condition_arg 在 128-dim 处截断，不再投影到 n_class
      - 128-dim 隐藏层送入 GammaMDNHead（输出原始值）
      - predict_out['out'] 由 pi_logits/alpha_raw/beta_raw 推断得到
      - 新增 predict_out['pi_logits'], predict_out['alpha_raw'], predict_out['beta_raw']
    """

    def __init__(self, *args, K=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.condition_arg_proj = self.condition_arg[:3]  # Linear(68,128)→ReLU→Dropout
        del self.condition_arg

        self.mdn_head = GammaMDNHead(
            in_dim=128, K=K, C=kwargs.get('n_class', 4),
        )

    def forward(self, local_input, distal_input, arg_feature):
        # --- 复用主干计算直到融合 ---
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

        # MDN head 输出原始值
        pi_logits, alpha_raw, beta_raw = self.mdn_head(hidden_128)
        predict_out['pi_logits'] = pi_logits    # [batch, K], raw
        predict_out['alpha_raw'] = alpha_raw    # [batch, K, C], raw
        predict_out['beta_raw'] = beta_raw      # [batch, K, C], raw

        # out 由 pi_logits/alpha_raw/beta_raw 推断（供 evaluation 兼容）
        inferred = gamma_mdn_predict_from_output(predict_out)
        predict_out['out'] = inferred['logits']

        if 'local_h1' in local_outs:
            predict_out['local_h1'] = local_outs['local_h1']
            predict_out['local_h2'] = local_outs['local_h2']

        return predict_out, None
```

### 3.3 文件位置

```text
MuRaL/models/
  ├── nb_model.py                  → NB v1 (不变)
  ├── nb_model_v2.py               → NB v2 (不变)
  ├── nb_model_v3.py               → NB v3 (不变)
  ├── dirichlet_mdn_model.py       → Dirichlet MDN (不变)
  ├── gamma_mdn_model.py           → ← 新增：Gamma MDN 系列模型
  │                                  (含 GammaMDNHead、
  │                                   Network3_ARG_condition_GammaMDN、
  │                                   gamma_mdn_predict_from_output、
  │                                   compute_mdn_uncertainty)
  └── losses.py                    → GammaMDNClassificationLoss (追加)
```

---

## 4. 损失函数

### 4.1 GammaMDNClassificationLoss

**核心公式**：

```text
pi         = softmax(pi_logits)
alpha      = softplus(alpha_raw) + ε      (shape, > 0)
beta       = softplus(beta_raw) + ε       (rate, > 0)
λ_{k,c}    = alpha_{k,c} / beta_{k,c}     (Gamma 期望)
p_k(c)     = λ_{k,c} / Σ_j λ_{k,j}        (组件 k 下的类概率)
p(y=c)     = Σ_k pi_k · p_k(c)
log p(y=c) = logsumexp(log pi_k + log p_k(c))
loss       = -log p(y)
```

**与用户提供的初始版本的关键改进**：

1. **数值稳定性**：使用 `log_alpha - log_beta` 计算 `log_lam`，再通过 `log_softmax` 获得 `log_p_k`，避免 `log(lam + 1e-8)` 的精度损失
2. **批量化**：用 gather/scatter 替换 Python for 循环，支持全 batch 并行
3. **reduction 参数**：支持 `'sum'`、`'mean'`、`'none'`
4. **支持 dict/tuple pred 格式**：与 Dirichlet MDN 一致的 unpack 接口

```python
class GammaMDNClassificationLoss(nn.Module):
    """Gamma-MDN classification loss.

    Model outputs raw pi_logits, alpha_raw, beta_raw; loss applies
    activation internally (log_softmax for pi, softplus for alpha/beta).

    Supports 2 pred formats:
      - dict: {'pi_logits': (B,K), 'alpha_raw': (B,K,C), 'beta_raw': (B,K,C)}
      - tuple: (pi_logits, alpha_raw, beta_raw)
    """

    def __init__(
        self,
        eps: float = 1e-8,
        reduction: str = 'sum',
    ):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, y):
        pi_logits, alpha_raw, beta_raw = self._unpack_pred(pred)
        B, K, C = alpha_raw.shape

        # --- 激活 ---
        log_pi = F.log_softmax(pi_logits, dim=1)           # (B, K)

        alpha = F.softplus(alpha_raw) + self.eps            # (B, K, C), shape
        beta = F.softplus(beta_raw) + self.eps              # (B, K, C), rate

        # --- log p_k(c) via numerically stable log_softmax over log(alpha/beta) ---
        log_lam = torch.log(alpha) - torch.log(beta)        # (B, K, C), log of Gamma mean
        log_p_k = F.log_softmax(log_lam, dim=-1)            # (B, K, C), log class probs per component

        # --- log p(y) via logsumexp ---
        y_idx = y.view(B, 1, 1).expand(B, K, 1)             # (B, K, 1)
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
            "pred should be dict with keys ['pi_logits', 'alpha_raw', 'beta_raw'] "
            "or tuple (pi_logits, alpha_raw, beta_raw)."
        )
```

### 4.2 LossFactory 集成

在 `losses.py` 的 `LossFactory.create_loss()` 中新增分支：

```python
elif loss_name == 'GammaMDN':
    return GammaMDNClassificationLoss(reduction='sum')
```

### 4.3 AdaptiveLossStrategy2 适配

在 `custom_loss.py` 的 `AdaptiveLossStrategy2.calc_loss()` 中新增分支：

```python
is_dir_mdn = isinstance(criterion, DirichletMDNClassificationLoss)
is_gamma_mdn = isinstance(criterion, GammaMDNClassificationLoss)

if is_dir_mdn or is_gamma_mdn:
    # MDN loss: preds dict 直接传给 criterion
    loss = criterion(preds, y)
    loss_local1 = loss_local2 = loss_local3 = None
    loss_mid = loss_distal = None
    loss_arg_feature = None
    loss_dual_head = 0
```

这样 Gamma MDN 与 Dirichlet MDN 共享同一分支：`_calc_loss` 中的 criterion 调用会直接接收整个 `preds` dict，由 loss 内部的 `_unpack_pred` 取出 `pi_logits`/`alpha_raw`/`beta_raw`。

---

## 5. 预测与不确定量化

预测推理和不确定量化拆分为两个独立函数，职责分离、各自聚焦。

### 5.1 预测推理

仅输出预测相关量，不包含不确定量化指标。

```python
def gamma_mdn_predict_from_output(out, eps=1e-8):
    """从 Gamma MDN 模型输出推断最终预测。

    out 应包含原始值 'pi_logits', 'alpha_raw', 'beta_raw'。
    函数内部完成激活（softmax, softplus）。
    仅返回预测相关量，不确定量化见 `compute_mdn_uncertainty`。
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
```

### 5.2 不确定量化

独立函数，仅计算不确定量化指标。后续可在此处拓展其他 uncertainty 计算方式（如 Dirichlet evidence、MC dropout 等）。

```python
def compute_mdn_uncertainty(predict_out, eps=1e-8):
    """从 MDN 模型输出计算不确定量化指标。

    接收 model forward 的 predict_out dict（含原始值），内部完成激活。
    当前仅支持 pi_entropy，后续可拓展。

    Args:
        predict_out: dict，需含 'pi_logits' (B, K)

    Returns:
        dict with uncertainty metrics, e.g.:
          {'pi_entropy': (B,) Tensor}
    """
    pi_logits = predict_out['pi_logits']          # (B, K), raw

    pi = F.softmax(pi_logits, dim=1)                           # (B, K)
    pi_entropy = -(pi * torch.log(pi + eps)).sum(dim=1)        # (B,)

    return {
        'pi_entropy': pi_entropy,
    }
```

---

## 6. 评估

用 `pi_entropy` 替代 evidence 作为不确定性指标。由于突变数据存在严重类不平衡（class 0 占绝大多数），评估重点不是 overall accuracy，而是 **每个 mutation type (prob1~prob3) 的 obs 密度与 pred 均值在各熵 bin 间的一致性**。

**核心思路**：按熵分 bin 后，对每个 prob c ∈ {1,2,3}：

```text
对于每个 bin b:
  obs_density_{c,b} = mean(y == c in bin b)
  pred_mean_{c,b}   = mean(prob_c in bin b)

correlation_c = corr(obs_density_c, pred_mean_c)  # 跨 bin 的 Pearson 相关
```

低熵（高置信度）的 bin 若有更好的 obs/pred 一致性，说明模型的置信度判断是有意义的。

### 6.1 GammaMDNEvaluator

```python
class GammaMDNEvaluator(Evaluator):
    """Evaluator 子类，按 pi_entropy 分 bin。

    由于类不平衡，不评估 overall accuracy，而是对每个 mutation type
    c ∈ {1,2,3} 计算跨 bin 的 obs 密度 vs pred 均值相关性。
    同时输出每个 bin 的明细 DataFrame。
    """

    def __init__(self, data_local, y_prob, n_class, pi_entropy=None, n_bins=10,
                 calibra=None, use_obs_count=False, printer=print):
        super().__init__(data_local, y_prob, n_class, calibra=calibra,
                         use_obs_count=use_obs_count, printer=printer)
        self.pi_entropy = pi_entropy
        self.n_bins = n_bins
        self.bin_results = None  # DataFrame with per-bin obs/pred info

    def evaluate_entropy_calibration(self):
        """按 pi_entropy 分 bin，计算每个 prob 的 obs 密度 vs pred 均值。"""
        if self.pi_entropy is None:
            return None

        pi_entropy_np = self.pi_entropy
        if isinstance(pi_entropy_np, torch.Tensor):
            pi_entropy_np = pi_entropy_np.numpy()

        bin_edges = np.percentile(pi_entropy_np,
            np.linspace(0, 100, self.n_bins + 1))

        true_label = self.data_and_prob['mut_type'].values
        n_probs = self.y_prob.shape[1]  # 含 prob0

        records = []
        for i in range(self.n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            if i == self.n_bins - 1:
                mask = (pi_entropy_np >= lo) & (pi_entropy_np <= hi)
            else:
                mask = (pi_entropy_np >= lo) & (pi_entropy_np < hi)

            n_samples = mask.sum()
            if n_samples == 0:
                continue

            avg_entropy = pi_entropy_np[mask].mean()
            label = (
                f"ent<{hi:.3f}" if i == 0 else
                f"ent>={lo:.3f}" if i == self.n_bins - 1 else
                f"ent=[{lo:.3f},{hi:.3f})"
            )

            record = {
                'bin': i,
                'bin_label': label,
                'n': n_samples,
                'entropy': avg_entropy,
            }

            for c in range(1, n_probs):  # prob1, prob2, ... (skip prob0)
                obs_density = (true_label[mask] == c).mean()
                pred_mean = self.y_prob[mask, c].mean()
                record[f'prob{c}_obs_density'] = obs_density
                record[f'prob{c}_pred_mean'] = pred_mean

            records.append(record)

        self.bin_results = pd.DataFrame(records)
        self._print_bin_results()

        # 跨 bin 计算每个 prob 的 obs 密度与 pred 均值的相关性
        correlations = {}
        for c in range(1, n_probs):
            obs_col = f'prob{c}_obs_density'
            pred_col = f'prob{c}_pred_mean'
            if obs_col in self.bin_results.columns and len(self.bin_results) >= 3:
                corr = self.bin_results[obs_col].corr(self.bin_results[pred_col])
                correlations[f'prob{c}_corr'] = corr
                self.printer(
                    f"  prob{c} obs vs pred correlation: {corr:.4f}"
                )
            else:
                correlations[f'prob{c}_corr'] = None

        return correlations

    def _print_bin_results(self):
        """打印每个 bin 的汇总信息。"""
        for _, row in self.bin_results.iterrows():
            parts = [
                f"  bin{row['bin']+1} {row['bin_label']}: "
                f"n={int(row['n']):>6d}  entropy={row['entropy']:.4f}"
            ]
            for c in range(1, self.y_prob.shape[1]):
                parts.append(
                    f"  prob{c}_obs={row[f'prob{c}_obs_density']:.6f}  "
                    f"prob{c}_pred={row[f'prob{c}_pred_mean']:.6f}"
                )
            self.printer(''.join(parts))

    def get_bin_results(self):
        """返回 bin 级 DataFrame。"""
        return self.bin_results
```

### 6.2 PiEntropyRecoder

与 `EvidenceRecoder` 类似，收集验证集的不确定量化指标。使用独立的 `compute_mdn_uncertainty` 函数，不依赖预测推理函数。

```python
class PiEntropyRecoder(Observer):
    """收集 MDN 模型输出的不确定量化指标（当前为 pi_entropy）。

    使用独立的 compute_mdn_uncertainty 函数，与预测推理解耦。
    """

    def __init__(self):
        super().__init__()
        self.pi_entropy = None

    def recode(self, preds):
        predict_out, _ = preds if isinstance(preds, tuple) else (preds, None)
        if 'pi_logits' not in predict_out:
            return
        with torch.no_grad():
            uncertainty = compute_mdn_uncertainty(predict_out)
            entropy = uncertainty['pi_entropy'].cpu()
        self.pi_entropy = (
        self.pi_entropy = (
            entropy if self.pi_entropy is None
            else torch.cat([self.pi_entropy, entropy], dim=0)
        )

    def output(self):
        val = self.pi_entropy
        self.reset()
        return val

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            self.recode(kwargs['valid_preds'])
```

### 6.3 训练脚本集成

```python
if is_gamma_mdn:
    valid_pi_entropy = trainer.get_pi_entropy()
    evaluator = GammaMDNEvaluator(data_local_valid, valid_y_prob, n_class,
                                   pi_entropy=valid_pi_entropy, printer=print)
    evaluator.evaluate_entropy_calibration()
else:
    evaluator = Evaluator(...)
```

---

## 7. model_config.py 注册

```python
elif model_no == '151_gamma_mdn':
    from MuRaL.models.gamma_mdn_model import Network3_ARG_condition_GammaMDN

    model_config = {
        'pooling_kind': 'max',
        'embeding_avg_mutations': False,
        'embeding_nuc_skew': False,
        'no_of_nuc_skew': 14,
        'use_local_fc2': True,
        'use_local_fc3': True,
        'local_model_name': 'AverageMutationModel_add2DCNN',
        'local_fc2_name': 'hidden_with_relu',
        'local_fc3_name': 'ConvModelDrop',
    }

    model_specify_config = {
        'fused_type': 'logit',
        'n_arg_features': 23,
        'arg_hidden_dim': 128,
        'arg_out_dim': 64,
        'arg_dropout': [0.2, 0.1, 0.1],
        'K': 3,  # Gamma MDN 组件数量
    }

    model_config.update(model_specify_config)

    model = Network3_ARG_condition_GammaMDN(
        emb_dims, ..., n_class=n_class, config=model_config,
    )
```

---

## 8. 实施步骤

| # | 任务 | 文件 | 说明 |
|---|---|---|---|
| 1 | 定义 `GammaMDNClassificationLoss` | `losses.py` | loss 类 + LossFactory 注册 `'GammaMDN'` |
| 2 | 定义 `GammaMDNHead` | `gamma_mdn_model.py` | MDN 头，输出原始 pi_logits / alpha_raw / beta_raw |
| 3 | 定义 `gamma_mdn_predict_from_output` | `gamma_mdn_model.py` | 内部激活，推断 prob/logits |
| 3b | 定义 `compute_mdn_uncertainty` | `gamma_mdn_model.py` | 内部激活，计算 pi_entropy 等不确定量化指标 |
| 4 | 定义 `Network3_ARG_condition_GammaMDN` | `gamma_mdn_model.py` | 继承 151，接入 GammaMDN head |
| 5 | 适配 `AdaptiveLossStrategy2` | `custom_loss.py` | 新增 `is_gamma_mdn` 分支（与 `is_dir_mdn` 共用） |
| 6 | 注册模型别名 | `model_config.py` | 添加 `151_gamma_mdn` |
| 7 | 训练脚本集成 | `trainingv2.py` | 参数校验（与 Dirichlet MDN 类似） |

---

## 9. 注意事项

1. **激活时机**：model forward 始终输出原始值（`pi_logits`, `alpha_raw`, `beta_raw`）。`GammaMDNClassificationLoss` 内部做 `softmax + softplus` 激活，`predict_from_output` 也做同样的激活。二者相互独立、不共享状态。

2. **`predict_out['out']` 无 `no_grad()`**：out 由 `predict_from_output` 推断得到，保留完整梯度路径。

3. **alpha 和 beta 均为 softplus + eps 激活**：Gamma 分布的 shape 和 rate 都必须 > 0。`softplus` 保证平滑可微，`eps` 防止除零。

4. **数值稳定性**：使用 `log_alpha - log_beta` → `log_softmax` 替代 `log(alpha/beta + eps)` → `nll_loss`，避免在 log 内部做归一化带来的精度损失。与原实现相比，这是重要的改进。

5. **与原实现的等价性验证**：

   原实现（用户提供）：
   ```python
   lam = alpha / beta                           # (B, 4)  per component
   lam = lam / lam.sum(dim=1, keepdim=True)     # normalize
   log_p = torch.log(lam + 1e-8)               # log prob
   nll = F.nll_loss(log_p, y, reduction='none')
   ```

   改进后实现（批量化 + 数值稳定）：
   ```python
   log_lam = torch.log(alpha) - torch.log(beta)          # (B, K, C)
   log_p_k = F.log_softmax(log_lam, dim=-1)              # (B, K, C)
   log_p_y = log_p_k.gather(dim=2, index=y_idx).squeeze(2)  # (B, K)
   log_likelihood = torch.logsumexp(log_pi + log_p_y, dim=1)
   ```

   两个实现在数学上等价，但改进后:
   - 避免 for 循环，利用 GPU 并行
   - `log_softmax` 比 `log(softmax)` 数值更稳定
   - 将 K 个 NLL 在 logsumexp 内合并，而非外置减法

6. **`reduction='sum'`**：默认与 CE/NB/DirichletMDN 一致。可通过参数切换。

7. **K 的选择**：默认 K=3，通过 config 的 `K` 控制。

8. **不对预训练权重做加载兼容**：`GammaMDN` 的 state_dict 与原始 151 不兼容（多了 `mdn_head.*` 参数，少了 `condition_arg.3` 参数）。

9. **不确定性通过 pi 熵衡量**：`pi_entropy = -Σ pi·log pi`。低熵 = 某个混合组件主导 → 高置信度；高熵 = 组件权重均匀 → 低置信度。该指标天然适用于 MDN 的 mixture 结构，且计算简单，不受 Gamma 参数尺度影响。

10. **预测与不确定量化职责分离**：`gamma_mdn_predict_from_output` 只负责预测推理（prob/logits），`compute_mdn_uncertainty` 只负责不确定量化。二者独立演化：后续可新增 MC dropout、ensemble variance 等 uncertainty 指标，不污染预测接口。

11. **后续可能拓展**：
    - 对 pi 熵加入 regularizer（鼓励低熵 → 更 sharp 的 mixture 权重）
    - 与 Dirichlet MDN 组合的 ensemble 模型

---

## 附录：设计决策记录

以下决策来自 grill-me 讨论，已确认并落地。

| # | 决策 | 结论 | 原因 |
|---|------|------|------|
| Q1 | GammaMDNHead hidden_dim | **64**（与 DirMDN 一致） | 避免与 DirMDN 对比时的 confounding factor；容量验证列入后续 TO-DO |
| Q2 | `compute_mdn_uncertainty` 内 `no_grad()` | **不做** | 函数纯计算，计算图管理由调用方决定；后续 uncertainty regularizer 可能需要梯度 |
| Q3 | Stochastic sampling | **不做** | 突变率之和不为 1，非严格分类任务，确定性 Gamma 期望归一化即可 |
| Q4 | K 默认值 | **3**，config 可调 | 与 DirMDN 一致，后续通过实验对比 K={2,3,5} |
| Q5 | n_bins | **10**（原 5） | 5 个 bin 对 Pearson 相关过于敏感，10 个更稳健 |
| Q6 | 训练监控指标 | **仅 total NLL** | 子分支 loss 均为 None，不提供分解 |
| Q7 | trainingv2.py 分支策略 | **`is_gamma_mdn` 独立分支** | 实验阶段便于独立调整；统一合并属于过早优化 |
| Q8 | `compute_mdn_uncertainty` 归属 | **`gamma_mdn_model.py`** | 当前仅 Gamma MDN 使用，后续若有通用需求再迁移 |
| Q9 | 等价性验证 | **已通过** | `test_gamma_mdn_loss_equivalence.py`，6 个测试全部通过 |
| Q10 | 正则化项 | **不加** | 基线版本先跑通，若 K 个组件退化（pi 均匀）再考虑 entropy reg |
| Q11 | `predict_out['out']` 格式 | **log-prob** | `torch.log(prob + eps)`，与 DirMDN 一致，评估管道兼容 |
