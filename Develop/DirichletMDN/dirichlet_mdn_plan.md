# Dirichlet MDN 拓展方案

## 概述

在 **model 151** (`Network3_ARG_condition`) 基础上引入 **Dirichlet Mixture Density Network (MDN)** 头，通过继承方式实现，类似 NB loss 的拓展模式。

**核心区别**：模型不再单独预测 `out`（logits），而是输出 `pi_logits`（混合权重 logits）和 `alpha_raw`（Dirichlet 浓度参数原始值），由它们推断出最终的预测概率和证据强度。

**关键设计原则**：model forward 始终输出原始值（`pi_logits`, `alpha_raw`），不做任何激活。激活分别由 loss 和 `predict_from_output` 各自负责，避免冗余。

---

## 1. DirichletMDNHead

输出原始值，不激活。激活由 loss 和 `predict_from_output` 各自独立完成。

### 架构

```text
in_dim (128)
    ↓
Shared Backbone: Linear(128, 64) → ReLU
    ↓
  ┌── pi_linear(64, K)       → pi_logits     [batch, K]       (raw)
  └── alpha_linear(64, K*C)  → alpha_raw     [batch, K, C]    (raw)
```

### 实现

```python
class DirichletMDNHead(nn.Module):
    """Dirichlet Mixture Density Network head.

    Architecture: in_dim → shared MLP → pi_head / alpha_head.
    Outputs raw values (no activation). Activation is handled by
    the loss function and predict_from_output independently.

    Args:
        in_dim: 输入特征维度
        K: Dirichlet 组件数量
        C: 类别数（默认 4）
        hidden_dim: 共享隐藏层维度（默认 64）
    """
    def __init__(self, in_dim, K=3, C=4, hidden_dim=64):
        super().__init__()
        self.K = K
        self.C = C

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden_dim, K)          # → pi_logits (raw)
        self.alpha = nn.Linear(hidden_dim, K * C)   # → alpha_raw (raw)

    def forward(self, x):
        h = self.backbone(x)                                  # [batch, hidden_dim]
        pi_logits = self.pi(h)                                # [batch, K]
        alpha_raw = self.alpha(h).view(-1, self.K, self.C)    # [batch, K, C]
        return pi_logits, alpha_raw
```

---

## 2. in_dim 连接位置

### 2.1 Model 151 架构回顾

```text
local_out, local2, local3      ← local_scale_model    ← 各 [batch, n_class=4]
distal_out1 (mid)               ← middle_scale_model   ← [batch, n_class=4]
distal_out2 (distal)            ← large_scale_model    ← [batch, n_class=4]
arg_feature                     ← arg_branch           ← [batch, arg_out_dim=64]

fusion_out = fuse(local, local2, local3, (mid+distal)/2)  ← [batch, n_class=4]

# condition_arg: Sequential 定义
#   [0] Linear(4+64, 128)  → [batch, 128]
#   [1] ReLU
#   [2] Dropout(0.1)
#   [3] Linear(128, 4)     → [batch, 4]  = out (不再使用)
```

### 2.2 候选方案对比

| 方案 | 位置 | 维度 | 获取方式 | 优点 | 缺点 |
|---|---|---|---|---|---|
| **A ⭐** | `condition_arg` 中间层 (ReLU 后) | **128** | 拆分 `condition_arg` | 信息最丰富；维度适合 MDN head | 需拆解原 Sequential |
| B | `concat([fusion_out, arg_feature])` | 68 (4+64) | 直接 cat | 无需改造模型 | 未经过深层处理；维度小于 128 |
| C | NBv3 风格：各子模型高维特征 → 融合 MLP | 各子模型 last Linear 的 in_features | forward hook | 保留子模型独立信息 | 复杂度高；与 NBv3 冗余 |

**推荐方案 A**：将 `condition_arg` 拆分为两段，暴露 128-dim 隐藏层作为 `DirichletMDNHead` 的输入。

#### 拆分方式

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
)  # → [batch, 128] 作为 MDN head 的输入
```

---

## 3. 模型类设计

### 3.1 继承结构

```text
Network3_ARG_condition (model 151)
  ├── Network3_ARG_condition_NB     (151_nb)       — NB v1
  ├── Network3_ARG_condition_NBv2   (151_nb_v2)    — NB v2
  ├── Network3_ARG_condition_NBv3   (151_nb_v3)    — NB v3
  └── Network3_ARG_condition_DirMDN (151_dir_mdn)  — Dirichlet MDN  <-- 新增
```

### 3.2 关键实现

```python
class Network3_ARG_condition_DirMDN(Network3_ARG_condition):
    """Model 151 variant with Dirichlet MDN head.

    与普通 151 的关键区别:
      - condition_arg 在 128-dim 处截断，不再投影到 n_class
      - 128-dim 隐藏层送入 DirichletMDNHead（输出原始值）
      - predict_out['out'] 由 pi_logits/alpha_raw 推断得到
      - 新增 predict_out['pi_logits'], predict_out['alpha_raw']
      - out 由 predict_from_output 计算（无 no_grad，后续可参与 loss）
    """

    def __init__(self, *args, K=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.condition_arg_proj = self.condition_arg[:3]  # Linear(68,128)→ReLU→Dropout
        del self.condition_arg

        self.mdn_head = DirichletMDNHead(
            in_dim=128, K=K, C=kwargs.get('n_class', 4)
        )

    def forward(self, local_input, distal_input, arg_feature):
        # --- 复用主干计算直到融合（同基类） ---
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
        pi_logits, alpha_raw = self.mdn_head(hidden_128)
        predict_out['pi_logits'] = pi_logits    # [batch, K], raw
        predict_out['alpha_raw'] = alpha_raw    # [batch, K, C], raw

        # out 由 pi_logits/alpha_raw 推断（供 evaluation 兼容，无 no_grad）
        inferred = dirichlet_mdn_predict_from_output(predict_out)
        predict_out['out'] = inferred['logits']

        if 'local_h1' in local_outs:
            predict_out['local_h1'] = local_outs['local_h1']
            predict_out['local_h2'] = local_outs['local_h2']

        return predict_out, None
```

### 3.3 变体：带 auxiliary CE head 的 Dirichlet MDN

**设计动机**：保留 `condition_arg_out` 作为 auxiliary head，与 MDN head 并行输出，给 128-dim 隐藏层一个额外的强监督信号。

```text
hidden_128 (128-dim)
    ├── condition_arg_out → out_aux    [batch, C]  ← CE auxiliary loss (λ=0.1)
    └── mdn_head          → pi_logits, alpha_raw   ← Dirichlet MDN loss (λ=1.0)
```

```python
class Network3_ARG_condition_DirMDN_aux(Network3_ARG_condition):
    """151 + Dirichlet MDN head + auxiliary CE head."""
    def __init__(self, *args, K=3, aux_loss_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_arg_proj = self.condition_arg[:3]
        self.condition_arg_out = self.condition_arg[3]
        del self.condition_arg
        self.mdn_head = DirichletMDNHead(in_dim=128, K=K, C=kwargs.get('n_class', 4))
        self.aux_loss_weight = aux_loss_weight

    def forward(self, local_input, distal_input, arg_feature):
        # ...（同 DirMDN，直到 hidden_128）...
        predict_out['out_aux'] = self.condition_arg_out(hidden_128)
        predict_out['out'] = predict_out['out_aux']
        pi_logits, alpha_raw = self.mdn_head(hidden_128)
        predict_out['pi_logits'] = pi_logits
        predict_out['alpha_raw'] = alpha_raw
        return predict_out, None
```

**模型别名**：`151_dir_mdn_aux`

**注**：aux 变体的 loss 计算需要独立 loss strategy（如 `AdaptiveLossStrategy3`），本次不实现，仅记录。

### 3.4 文件位置

```text
MuRaL/models/
  ├── nb_model.py              → NB v1 (不变)
  ├── nb_model_v2.py           → NB v2 (不变)
  ├── nb_model_v3.py           → NB v3 (不变)
  ├── dirichlet_mdn_model.py   → ← 新增：Dirichlet MDN 系列模型
  │                               (含 DirichletMDNHead、
  │                                Network3_ARG_condition_DirMDN、
  │                                Network3_ARG_condition_DirMDN_aux、
  │                                dirichlet_mdn_predict_from_output)
  └── losses.py                → ← 新增：DirichletMDNClassificationLoss
```

---

## 4. 损失函数

### 4.1 DirichletMDNClassificationLoss

Loss 内部完成激活（`softmax` for pi, `softplus` for alpha），model 只提供原始值。

#### 核心公式

```text
pi         = softmax(pi_logits)
alpha      = softplus(alpha_raw) + ε
p(y=c)     = Σₖ piₖ · alpha_{k,c} / sum_j(alpha_{k,j})
log p(y=c) = logsumexp( log piₖ + log p_k(c) )
loss       = -log p(y) + λ₁ · evidence_reg + λ₂ · (-entropy_reg)
```

#### 实现

```python
class DirichletMDNClassificationLoss(nn.Module):
    """Dirichlet-MDN classification loss.

    支持 2 种 pred 格式:
      - dict: {'pi_logits': (B,K), 'alpha_raw': (B,K,C)}
      - tuple: (pi_logits, alpha_raw)

    Loss 内部自行完成 pi_logits → pi, alpha_raw → alpha 的激活。
    """

    def __init__(
        self,
        eps: float = 1e-8,
        evidence_reg: float = 0.0,
        entropy_reg: float = 0.0,
        reduction: str = "sum",
    ):
        super().__init__()
        self.eps = eps
        self.evidence_reg = evidence_reg
        self.entropy_reg = entropy_reg
        self.reduction = reduction

    def forward(self, pred, y):
        pi_logits, alpha_raw = self._unpack_pred(pred)
        B, K, C = alpha_raw.shape

        # --- 激活 ---
        log_pi = F.log_softmax(pi_logits, dim=1)           # (B, K), log_softmax 数值稳定
        pi = log_pi.exp()                                   # (B, K), 精确 softmax
        alpha = F.softplus(alpha_raw) + self.eps            # (B, K, C)

        # --- log p(y) via logsumexp ---
        alpha_sum = alpha.sum(dim=-1, keepdim=True)         # (B, K, 1)
        log_p_k = torch.log(alpha + self.eps) - torch.log(alpha_sum + self.eps)

        y_idx = y.view(B, 1, 1).expand(B, K, 1)             # (B, K, 1)
        log_p_y = log_p_k.gather(dim=2, index=y_idx).squeeze(2)  # (B, K)

        log_likelihood = torch.logsumexp(log_pi + log_p_y, dim=1)  # (B,)
        nll_loss = -log_likelihood

        # --- reduction ---
        if self.reduction == "mean":
            nll_loss = nll_loss.mean()
        elif self.reduction == "sum":
            nll_loss = nll_loss.sum()
        elif self.reduction != "none":
            raise ValueError(f"Unknown reduction: {self.reduction}")

        loss = nll_loss

        # --- evidence regularizer (尺度与 NLL 对齐) ---
        if self.evidence_reg > 0:
            evidence = alpha.sum(dim=-1)                    # (B, K)
            if self.reduction == "sum":
                reg = evidence.sum() / K                    # sum, 尺度 ~O(B)
            else:
                reg = evidence.mean()                       # mean, 尺度 ~O(1)
            loss = loss + self.evidence_reg * reg

        # --- entropy regularizer (尺度与 NLL 对齐) ---
        if self.entropy_reg > 0:
            per_sample_entropy = -(pi * log_pi).sum(dim=1)                     # (B,)
            if self.reduction == "sum":
                entropy = per_sample_entropy.sum()          # sum, 尺度 ~O(B)
            else:
                entropy = per_sample_entropy.mean()         # mean, 尺度 ~O(1)
            loss = loss - self.entropy_reg * entropy

        return loss

    @staticmethod
    def _unpack_pred(pred):
        if isinstance(pred, dict):
            return pred["pi_logits"], pred["alpha_raw"]
        if isinstance(pred, (tuple, list)):
            if len(pred) != 2:
                raise ValueError("Tuple pred should be (pi_logits, alpha_raw).")
            return pred[0], pred[1]
        raise TypeError(
            "pred should be dict with keys ['pi_logits', 'alpha_raw'] "
            "or tuple (pi_logits, alpha_raw)."
        )
```

### 4.2 LossFactory 集成

```python
elif loss_name == 'DirichletMDN':
    return DirichletMDNClassificationLoss(
        reduction='sum',
        evidence_reg=0.0,   # 由 config 传入
        entropy_reg=0.0,
    )
```

`evidence_reg` 和 `entropy_reg` 由训练脚本从 config 读取后传入（路径 B）。

### 4.3 AdaptiveLossStrategy2 适配

```python
is_dir_mdn = isinstance(criterion, DirichletMDNClassificationLoss)
if is_dir_mdn:
    loss = criterion(preds, y)  # preds 即 predict_out dict, 含 pi_logits/alpha_raw
    loss_local1 = loss_local2 = loss_local3 = None
    loss_mid = loss_distal = None
    loss_arg_feature = None
    loss_dual_head = 0
```

---

## 5. 预测推理

```python
def dirichlet_mdn_predict_from_output(out, eps=1e-8):
    """从 Dirichlet MDN 模型输出推断最终预测。

    out 应包含原始值 'pi_logits' 和 'alpha_raw'。
    函数内部完成激活（softmax, softplus）。
    """
    pi_logits = out["pi_logits"]       # (B, K), raw
    alpha_raw = out["alpha_raw"]       # (B, K, C), raw

    pi = F.softmax(pi_logits, dim=1)                              # (B, K)
    alpha = F.softplus(alpha_raw) + eps                           # (B, K, C)

    p_k = alpha / alpha.sum(dim=-1, keepdim=True)                 # (B, K, C)
    prob = (pi.unsqueeze(-1) * p_k).sum(dim=1)                    # (B, C)

    evidence_k = alpha.sum(dim=-1)                                # (B, K)
    evidence = (pi * evidence_k).sum(dim=1)                       # (B,)

    return {
        "prob": prob,
        "logits": torch.log(prob + eps),
        "pred_class": prob.argmax(dim=-1),
        "pi": pi,
        "alpha": alpha,
        "p_k": p_k,
        "evidence": evidence,
        "uncertainty": 1.0 / (evidence + eps),
    }
```

---

## 6. 评估：DirMDNEvaluator

按 evidence 分 bin 评估 calibration 质量。低 evidence 的样本预期校准更差。

```python
class DirMDNEvaluator(Evaluator):
    """Evaluator 子类，按 evidence 分 bin 评估 calibration。"""

    def __init__(self, data_local, y_prob, n_class, evidence=None, n_bins=5,
                 calibra=None, use_obs_count=False, printer=print):
        super().__init__(data_local, y_prob, n_class, calibra=calibra,
                         use_obs_count=use_obs_count, printer=printer)
        self.evidence = evidence
        self.n_bins = n_bins

    def evaluate_evidence_calibration(self):
        """按 evidence 分 bin，报告每桶的准确率、置信度、证据均值。"""
        if self.evidence is None:
            return

        bin_edges = np.percentile(self.evidence,
            np.linspace(0, 100, self.n_bins + 1))

        true_label = self.data_and_prob['mut_type'].values
        pred_class = self.y_prob.argmax(axis=1)
        max_prob = self.y_prob.max(axis=1)

        for i in range(self.n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]
            mask = (self.evidence >= lo) & (self.evidence <= hi)
            if i == 0:
                # 第一桶包含下界相等值
                pass
            elif i == self.n_bins - 1:
                mask = (self.evidence >= lo) & (self.evidence <= hi)
            else:
                mask = (self.evidence >= lo) & (self.evidence < hi)

            if mask.sum() == 0:
                continue

            acc = (pred_class[mask] == true_label[mask]).mean()
            avg_conf = max_prob[mask].mean()
            avg_ev = self.evidence[mask].mean()

            label = (
                f"ev<{hi:.2f}" if i == 0 else
                f"ev>={lo:.2f}" if i == self.n_bins - 1 else
                f"ev=[{lo:.2f},{hi:.2f})"
            )
            self.printer(
                f"  bin{i+1} {label}: "
                f"n={mask.sum():>6d}  "
                f"acc={acc:.4f}  "
                f"conf={avg_conf:.4f}  "
                f"evidence={avg_ev:.2f}"
            )
```

### Evidence 收集（Trainer 内）

```python
class EvidenceRecoder(Observer):
    """收集 predict_out 中的 pi_logits/alpha_raw → 计算 evidence。"""

    def __init__(self):
        super().__init__()
        self.evidence = None

    def recode(self, preds):
        predict_out, _ = preds if isinstance(preds, tuple) else (preds, None)
        if 'pi_logits' not in predict_out:
            return
        with torch.no_grad():
            result = dirichlet_mdn_predict_from_output(predict_out)
            ev = result['evidence'].detach().cpu()
        self.evidence = (
            ev if self.evidence is None
            else torch.cat([self.evidence, ev], dim=0)
        )

    def output(self):
        ev = self.evidence
        self.reset()
        return ev

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            self.recode(kwargs['valid_preds'])
```

关键细节：
- 使用 `with torch.no_grad()` 确保 evidence 计算不保留计算图
- `.detach().cpu()` 及时释放 GPU 显存

Trainer 中通过 `collect_evidence=True` 控制注册/反注册，暴露 `get_evidence()` 方法。

### 训练脚本集成

```python
if is_dir_mdn:
    valid_evidence = trainer.get_evidence()
    evaluator = DirMDNEvaluator(data_local_valid, valid_y_prob, n_class,
                                evidence=valid_evidence, printer=print)
    evaluator.evaluate_evidence_calibration()
else:
    evaluator = Evaluator(...)
```

---

## 7. model_config.py 注册

```python
elif model_no == '151_dir_mdn':
    from MuRaL.models.dirichlet_mdn_model import Network3_ARG_condition_DirMDN

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
        'K': 3,  # Dirichlet MDN 组件数量
    }

    model_config.update(model_specify_config)

    model = Network3_ARG_condition_DirMDN(
        emb_dims, ..., n_class=n_class, config=model_config,
    )
```

---

## 8. 实施步骤

| # | 任务 | 文件 | 说明 |
|---|---|---|---|
| 1 | 定义 `DirichletMDNClassificationLoss` | `losses.py` | loss 类 + LossFactory 注册 |
| 2 | 定义 `DirichletMDNHead` | `dirichlet_mdn_model.py` | MDN 头，输出原始 pi_logits / alpha_raw |
| 3 | 定义 `dirichlet_mdn_predict_from_output` | `dirichlet_mdn_model.py` | 内部激活，推断 prob/logits/evidence |
| 4 | 定义 `Network3_ARG_condition_DirMDN` | `dirichlet_mdn_model.py` | 继承 151，接入 MDN head |
| 5 | 定义 `DirMDNEvaluator` | `trainingv2.py` | 继承 Evaluator，按 evidence 分 bin |
| 6 | 定义 `EvidenceRecoder` | `observer.py` | Observer 子类，collect evidence + detach.cpu |
| 7 | 适配 `AdaptiveLossStrategy2` | `losses.py` | 新增 `is_dir_mdn` 分支 |
| 8 | Trainer 集成 evidence 收集 | `train.py` | `collect_evidence` flag + `get_evidence()` |
| 9 | 注册模型别名 | `model_config.py` | 添加 `151_dir_mdn` |
| 10 | 训练脚本集成 | `trainingv2.py` | 参数校验 + DirMDNEvaluator 调用 |

---

## 9. 注意事项

1. **激活时机**：model forward 始终输出原始值（`pi_logits`, `alpha_raw`）。`DirichletMDNClassificationLoss` 内部做 `softmax + softplus` 激活，`predict_from_output` 也做同样的激活。二者相互独立、不共享状态。

2. **`predict_out['out']` 无 `no_grad()`**：out 由 `predict_from_output` 推断得到，保留完整梯度路径。当前 loss 直接从 `pi_logits`/`alpha_raw` 计算，不经过 `out`；但如果后续需要基于 `out` 计算额外的 loss（如 aux CE），梯度路径不会被阻断。

3. **正则项尺度对齐**：`reduction='sum'` 时 NLL 为 O(B)。`evidence_reg` 和 `entropy_reg` 的正则项相应调整：sum 模式下使用 `sum()` 而非 `mean()` 匹配 NLL 的尺度。

4. **alpha 不做 clamp**：不通过硬截断限制 alpha 值。`evidence_reg` 是控制 alpha 膨胀的正则化手段，更为合理。`softplus + eps` 已经保证了 alpha > 0。

5. **EvidenceRecoder 及时 detach**：验证时收集 evidence 使用 `detach().cpu()` 并包裹 `torch.no_grad()`，不保留计算图。

6. **`reduce`**：默认 `'sum'`，与 CE/NB 一致。可通过参数切换为 `'mean'` 或 `'none'`。

7. **K 的选择**：默认 K=3，通过 config 的 `K` 控制（路径 A）。

8. **正则化超参传入**：`evidence_reg` 和 `entropy_reg` 由训练脚本从 config 读取后传入 LossFactory（路径 B）。

9. **模型别名**：`151_dir_mdn`（标准），`151_dir_mdn_aux`（aux 变体，后续实现）。

10. **不对预训练权重做加载兼容**：`DirMDN` 的 state_dict 与原始 151 不兼容。
