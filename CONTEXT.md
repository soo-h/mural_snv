# MuRaL — Domain Glossary

## Core Entities

- **Site** (位点) — A single genomic position from the input BED file. The atomic unit of prediction. Each site carries local sequence context, distal encoding, functional annotations, and a mutation-type label.

- **Encoding Window** (编码窗口) — A contiguous genomic region whose length is defined by `segment_center`. Multiple sites falling within the same window share a single reference-sequence lookup for distal encoding, reducing I/O. One `__getitem__` call on the dataset returns one encoding window's worth of sites and features.

- **Window Group** (窗口组) — A collection of `windows_per_group` encoding windows whose sites are concatenated into a flat pool. Formed by merging consecutive encoding windows.

- **Site Pool** (位点池) — The flattened collection of all sites from a single window group. Training batches are drawn from a site pool via shuffling. The same thing as a window group, viewed from the sampling perspective.

- **Training Batch** (训练批次) — A set of `batch_size` (typically 256) sites sampled from a site pool. The unit consumed by `trainer.train_step()`.

## Data Flow

```
encoding windows  →  window groups (site pools)  →  training batches
```

1. Sites are partitioned into **encoding windows** by genomic proximity (shared distal context).
2. Consecutive encoding windows are merged into **window groups** to form site pools large enough for batch sampling.
3. **Training batches** are drawn from each site pool with shuffling.
4. An incomplete last batch **carries over** to the next site pool (`window_group carry-over`).
