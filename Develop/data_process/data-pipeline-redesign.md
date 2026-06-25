# 数据管道重新设计

2026-06-25 · 基于 [data-pipeline-review-20260624](../architecture-review/data-pipeline-review-20260624.html) 的评审结论

## 术语

| 术语 | 英文 | 含义 |
|---|---|---|
| 位点 | site | BED 文件中单个基因组位置，预测的基本单元 |
| 编码窗口 | encoding_window | `segment_center` 定义的连续基因组区间，多个 site 共享同一个远端编码上下文以减少 I/O |
| 窗口组 | window_group | `windows_per_group` 个 encoding_window 的 site 合并后的平层集合 |
| 位点池 | site_pool | 等同于 window_group，从采样视角的称呼 |
| 训练批次 | training_batch | 从 site_pool 中 shuffle 采样得到的 `batch_size` 个 site |
| 跨组拼接 | carry-over | 当 site_pool 末尾不足一个 batch 时，剩余 site 带入下一个 site_pool |

## 当前状态

### 三层结构

```
CombinedDatasetNPv2
  ↓ __getitem__ 返回 window-level dict
encoding_window DataLoader  (batch_size=1, shuffle=True, num_workers=N)
  ↓ 逐个产出 encoding_window
get_seg_share_dataset  (合并 windows_per_group 个 window → Create_DatasetSegment)
  ↓ 产出 site_pool (平层 Dataset)
batch DataLoader  (batch_size=256, shuffle=True, num_workers=0)
  ↓ 产出 training_batch
trainer.train_step()
```

### 问题

1. **每 epoch 重建**：`generate_data_batches()` 在每 epoch 末尾被重新调用（trainingv2.py:429-430），生成器耗尽后需要重新创建
2. **无意义的 DataLoader 层**：内层 batch DataLoader 的 `num_workers=0`，没有并行收益但增加复杂度
3. **三个变体**：`generate_data_batches`、`_v2`（已废弃）、`_filt`（已搁置），逻辑重叠
4. **术语混乱**："segment" 同时指 encoding_window、window_group、site_pool 三个概念
5. **上帝参数 `calc_loss_strategy_name`**：同时控制 batch 解包、模型调用、loss 计算、预测校验 4 件事

## 目标设计

### 两层结构

```
EncodingWindowDataset
  ↓ __getitem__ 返回 window-level dict
encoding_window_loader = DataLoader(
    window_dataset,
    batch_size=1,
    shuffle=shuffle_windows,
    num_workers=N,
    collate_fn=unwrap_batch,
)
  ↓ 逐个产出 encoding_window (dict)
SiteShuffleBuffer(
    window_iter=encoding_window_loader,
    site_batch_size=256,
    shuffle_buffer_size=10000,
    shuffle_sites=True,
    drop_last=False,
    feature_spec=FeatureBatchSpec(...),
)
  ↓ 产出 training_batch (tuple)
trainer.train_step()
```

### EncodingWindowDataset

替代 `CombinedDatasetNPv2`。接口不变——`__getitem__(index)` 返回 window-level dict。内部通过 `FeatureFactory` 管理 feature source。

```python
class EncodingWindowDataset(Dataset):
    def __init__(self, segments, features, features_without_train=['local_seq']):
        ...

    def __getitem__(self, index):
        features = {}
        for name in self.features:
            if name in self.features_without_train:
                continue
            features[name] = self.features[name].get(index)
        return features

    def __len__(self):
        return len(self.segments)
```

### unwrap_batch collate_fn

`encoding_window_loader` 的 `batch_size=1`，`default_collate` 会给每个 feature 加一维 `[1, ...]`。`unwrap_batch` 去掉这层：

```python
def unwrap_batch(batch):
    """batch_size=1 时，返回单个 dict 而非堆叠的 tensor。"""
    return batch[0]
```

### FeatureBatchSpec

定义 batch tuple 中 feature 的顺序契约。放在 `MuRaL/data/dataset.py`。是数据侧的单一真相源。

```python
class FeatureBatchSpec:
    """定义 batch tuple 中 feature 的顺序契约。"""

    required_keys = ["mut_type", "cat_x", "distal_x"]

    optional_key_order = [
        "step_avg_mut",
        "segment_avg_kmer_mut",
        "arg_feature",
        "nuc_skew",
        "segment_id_label",  # 历史策略记录，当前未使用
        "sample_weight",
    ]

    def __init__(self, enabled_optional_keys=None):
        self.enabled_optional_keys = enabled_optional_keys

    def get_feature_order(self, available_keys=None):
        keys = list(self.required_keys)
        for k in self.optional_key_order:
            if self.enabled_optional_keys is not None and k not in self.enabled_optional_keys:
                continue
            if available_keys is not None and k not in available_keys:
                continue
            keys.append(k)
        return keys
```

**构造时机**：从 `feature_config['features']` 的 keys 派生（数据侧驱动），不依赖 `calc_loss_strategy_name`。

### SiteShuffleBuffer

核心新增类。替代 `get_seg_share_dataset` + `Create_DatasetSegment` + 内层 batch DataLoader。

**内部 buffer 结构**（方案 A，等价于当前实现）：

```python
self.buffer = {
    'mut_type': tensor,       # shape: [n_buffered_sites]
    'cat_x': tensor,          # shape: [n_buffered_sites, cat_n]
    'distal_x': tensor,       # shape: [n_buffered_sites, 4, distal_len]
    # ... 其他 feature
}
```

每来一个 encoding_window（dict of tensors），把每个 feature 沿 site 维度 cat 到 buffer。

**工作模式**：带 carry-over 的滑动窗口

1. 从 `encoding_window_loader` 逐个获取 encoding_window
2. 将 window 内 site 展开（flatten）放入内部 buffer
3. buffer 满 `shuffle_buffer_size` 时，shuffle → 切 batch → yield
4. 不足一个 batch 的剩余 site 保留在 buffer 中（carry-over），与下一批 window 的 site 混合
5. epoch 结束时，flush buffer 中所有剩余 site

**输出格式**：按 `FeatureBatchSpec.get_feature_order()` 组装的 tuple，与当前 `dict_to_tuple_collate` 产出等价。

**两级 shuffle**：

| 层级 | 位置 | 参数 | 默认值 |
|---|---|---|---|
| window 级 | encoding_window_loader | `shuffle` | True |
| site 级 | SiteShuffleBuffer | `shuffle_sites` | True |

两级都开启时接近 site 级完全随机打乱。可根据 I/O 性能评估关闭 window 级 shuffle。

**关键参数**：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `window_iter` | 上游 encoding_window 迭代器 | — |
| `site_batch_size` | 训练 batch 大小 | 256 |
| `shuffle_buffer_size` | buffer 容量（site 数） | 10000 |
| `shuffle_sites` | 是否在 buffer 内 shuffle | True |
| `drop_last` | 是否丢弃最后不完整 batch | False |
| `feature_spec` | FeatureBatchSpec 实例 | — |

**carry-over 行为**（`drop_last=False`）：

```
buffer flush 1: [9500 sites] → yield 37 batches, 保留 64 sites
buffer flush 2: [64 + 9936 = 10000 sites] → yield 39 batches, 保留 16 sites
...
epoch end:     flush 保留的 site → yield 最后一批（可能 < batch_size）
```

### 训练循环变化

**当前**（trainingv2.py）：

```python
# 初始化
dataloader_train = generate_data_batches(encoding_window_loader, ...)

for epoch in range(epochs):
    trainer.train_step(dataloader_train)
    ...
    # 每 epoch 重建
    dataloader_train = generate_data_batches(encoding_window_loader, ...)
```

**目标**：

```python
# 初始化 — 只需一次
site_batch_iter = SiteShuffleBuffer(
    window_iter=encoding_window_loader,
    site_batch_size=batch_size,
    feature_spec=feature_spec,
    ...
)

for epoch in range(epochs):
    trainer.train_step(site_batch_iter)  # 可重用，自动重置
    ...
    # 无需重建
```

## 消费侧重构

### FeatureBatchSpec 与 BatchConfig 的关系

**思路 B：FeatureBatchSpec 为根，BatchConfig 引用它。**

```python
@dataclass
class BatchConfig:
    feature_spec: FeatureBatchSpec
    include_avg_mut_in_inputs: bool = False
    include_kmer_mut_in_inputs: bool = False
    # ... 其他 include_* 标志
```

`FeatureBatchSpec` 定义生成哪些 feature + 顺序（生产侧契约），`BatchConfig` 定义如何使用 feature（消费侧行为）。二者不分离——feature 生成但模型不用是浪费，模型要用但 feature 未生成是 bug。

### model_train_register_v2

替代 `model_train_register(strategy)` + 7 个 `model_train_*` 变体。基于 `FeatureBatchSpec` 解析，自动检测模型是 legacy 还是 modern。

```python
from MuRaL.models.nn_models import Network0, Network1, Network2

_LOCAL_INPUT_KEY_MAP = {
    'segment_avg_mut': 'avg_mutations',
    'step_avg_mut': 'avg_mutations',
    'segment_avg_kmer_mut': 'segment_avg_kmer_mut',
    'nuc_skew': 'nuc_skew',
}

_POSITIONAL_FEATURES = ['arg_feature']


def model_train_v2(inputs, model, spec):
    """统一的模型调用，基于 FeatureBatchSpec 解析。"""
    cont_x, cat_x, distal_x = inputs[0], inputs[1], inputs[2]

    # legacy 模型只接收 tuple (cont_x, cat_x)
    if isinstance(model, (Network0, Network1, Network2)):
        return model((cont_x, cat_x), distal_x)

    # modern 模型接收 dict local_input
    local_input = {
        'cont_data': cont_x,
        'cat_data': cat_x,
    }
    extra_positional = []

    optional_keys = [
        k for k in spec.get_feature_order()
        if k not in ('mut_type', 'cat_x', 'distal_x', 'sample_weight')
    ]
    for key, value in zip(optional_keys, inputs[3:]):
        if key in _LOCAL_INPUT_KEY_MAP:
            local_input[_LOCAL_INPUT_KEY_MAP[key]] = value
        elif key in _POSITIONAL_FEATURES:
            extra_positional.append(value)

    return model(local_input, distal_x, *extra_positional)


def model_train_register_v2(spec):
    """返回绑定到 FeatureBatchSpec 的 model_train 函数。"""
    def _train(inputs, model):
        return model_train_v2(inputs, model, spec)
    return _train
```

**对照审查**：

| 场景 | 旧版 | 新版 |
|---|---|---|
| Network0 + segment_soft_label | `model((cont_x, cat_x), distal_x)` | 检测到 Network0 → `model((cont_x, cat_x), distal_x)` ✓ |
| MuRaL_Network3 + AvgSegMutUseInLocal | `model({cont_data, cat_data, avg_mutations}, distal_x)` | 检测到 modern → 同上 ✓ |
| MuRaL_Network3 + SKA_local | `model({..., segment_avg_kmer_mut}, distal_x, arg_feature)` | 同上 ✓ |

### get_inputs_labels_v2

**部分重构**：batch 解包改为 spec 驱动；labels 构建仍由 strategy 控制（保留 strategy → labels 通路）。

```python
def get_inputs_labels_v2(batch, spec, strategy=None):
    """batch 解包改为 spec 驱动；labels 构建仍由 strategy 控制。"""
    # 用 spec 解包 batch
    feature_order = spec.get_feature_order()
    features = dict(zip(feature_order, batch))

    y = features['mut_type']
    cat_x = features['cat_x']
    distal_x = features['distal_x']
    sample_weight = features.get('sample_weight')

    # labels 构建仍由 strategy 控制（保留旧逻辑）
    config = STRATEGY_CONFIGS.get(strategy) if strategy else None
    labels = {'label': _process_label(y)}
    inputs = [0, cat_x, distal_x]

    if config:
        if config.include_avg_mut_in_labels and 'segment_avg_mut' in features:
            labels['avg_mut'] = features['segment_avg_mut']
        if config.include_kmer_mut_in_labels and 'segment_avg_kmer_mut' in features:
            labels['avg_kmer_mut'] = features['segment_avg_kmer_mut']
        if config.include_avg_mut_in_inputs and 'segment_avg_mut' in features:
            inputs.append(features['segment_avg_mut'])
        if config.include_kmer_mut_in_inputs and 'segment_avg_kmer_mut' in features:
            inputs.append(features['segment_avg_kmer_mut'])

    return labels, inputs, sample_weight
```

## 清理项

| 项 | 说明 |
|---|---|
| `generate_data_batches` | 被 `SiteShuffleBuffer` 替代 |
| `generate_data_batches_v2` | 已注释，删除 |
| `generate_data_batches_filt` | 仅 trainingBysegment.py（已搁置）使用，删除 |
| `get_seg_share_dataset` | 逻辑合并到 SiteShuffleBuffer |
| `Create_DatasetSegment` | 不再需要 |
| `Create_DatasetSegment_Adaptive` | 不再需要 |
| `MultiSegmentDatasetIterator` | 不再需要 |
| `dict_to_tuple_collate` | 被 `unwrap_batch` + SiteShuffleBuffer 内部组装替代 |
| `model_train_register`（旧版） | 保留，方便对照审查新版本 |
| `model_train_*` 变体（7 个） | 被 `model_train_v2` 替代，旧版保留 |
| `CombinedDatasetNPv2` | 先保留，EncodingWindowDataset 验证后移除 |

## 迁移范围

- **首批**：`trainingv2.py`、`run_predict.py`
- **后续**：`training.py`、`training_accumulation.py`、`training_ensemble.py`
- **不迁移**：`bk/` 备份文件、`trainingBysegment.py`（已搁置）

## 验证和预测

| 场景 | `shuffle_windows` | `shuffle_sites` | `drop_last` | carry-over | 特殊需求 |
|---|---|---|---|---|---|
| 训练 | True | True | False | 需要 | sample_weight |
| 验证 | False | False | False | 需要 | sample_weight |
| 预测 | False | False | False | 需要，且不能丢弃任何 site | 顺序严格对齐，无 sample_weight |

预测时 `shuffle_sites=False` 保证 buffer 中 site 保持到达顺序；`shuffle_windows=False` 保证 encoding_window 按基因组顺序到达。carry-over 改变 batch 边界但不改变总顺序，与 `data_local_test` 对齐。`sample_weight` 由 `FeatureBatchSpec` 的 `enabled_optional_keys` 控制。

## epoch 重用机制

```python
class SiteShuffleBuffer:
    def __iter__(self):
        """每 epoch 返回新的 batch 生成器。"""
        self._reset_buffer()
        return self._generate_batches()

    def _generate_batches(self):
        for window in self.window_iter:  # 隐式 iter(encoding_window_loader)
            self._append(window)
            if self.n_buffered >= self.shuffle_buffer_size:
                yield from self._flush()
        yield from self._flush(force_last=True)
```

`trainer.train_step(site_batch_iter)` 调用 `for batch in site_batch_iter:` → `iter(site_batch_iter)` → `SiteShuffleBuffer.__iter__` → 新生成器 → 遍历 encoding_window_loader → 耗尽 → 返回。下一 epoch 重复。carry-over 跨 flush 周期但不跨 epoch：`force_last=True` 时 flush 所有剩余 site。

## 暂缓项

- **候选 3**：FeatureFactory 配置驱动接口 — FeatureSource 协议已存在，消费侧（collate / 模型输入）的解耦留作后续优化。
- **`calc_loss_strategy_name` 与 labels 构建的解耦**：当前 `get_inputs_labels_v2` 中 labels 构建仍由 strategy 控制（`include_*_in_labels` 等），与 loss 计算共用 strategy 名。后续可考虑将 labels 构建也改为 spec 驱动。
