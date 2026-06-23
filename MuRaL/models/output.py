"""
PredictOutput and SegmentOutput dataclasses — the single model output format.

All models return Tuple[PredictOutput, Optional[SegmentOutput]].
Downstream code can read named fields via .get() / [] / .field for
backwards compatibility during migration.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


class _DictCompat:
    """Mixin that gives a dataclass dict-like access (__getitem__, get, items, etc.).

    Works alongside dataclass fields.  Fields named _* are excluded from
    iteration.  Uses a private sentinel so that a field whose value really is
    None can be distinguished from a key that does not exist.
    """

    _sentinel = object()

    def get(self, key, default=None):
        v = getattr(self, key, self._sentinel)
        return default if v is self._sentinel else v

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        object.__setattr__(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None

    def items(self):
        for k in self._field_names:
            v = getattr(self, k)
            if v is not None:
                yield k, v

    def keys(self):
        for k in self._field_names:
            if getattr(self, k) is not None:
                yield k

    def __iter__(self):
        return self.keys()


@dataclass
class PredictOutput(_DictCompat):
    """Sub-model logits and specialised head outputs.

    Every field defaults to None — models set only what they produce.
    """

    # core sub-model logits
    local: Optional[torch.Tensor] = None
    local2: Optional[torch.Tensor] = None
    local3: Optional[torch.Tensor] = None
    mid: Optional[torch.Tensor] = None
    distal: Optional[torch.Tensor] = None
    out: Optional[torch.Tensor] = None

    # optional / auxiliary
    arg_feature: Optional[torch.Tensor] = None
    local_h1: Optional[torch.Tensor] = None
    local_h2: Optional[torch.Tensor] = None

    # NB variant
    mu: Optional[torch.Tensor] = None
    r: Optional[torch.Tensor] = None

    # GammaMDN / DirMDN variant
    pi_logits: Optional[torch.Tensor] = None
    alpha_raw: Optional[torch.Tensor] = None
    beta_raw: Optional[torch.Tensor] = None

    # Gamma-Total-Dirichlet variant
    gamma_alpha_raw: Optional[torch.Tensor] = None
    gamma_beta_raw: Optional[torch.Tensor] = None
    dir_alpha_raw: Optional[torch.Tensor] = None

    # GAN variant
    construct_loss: Optional[torch.Tensor] = None

    _field_names: list = field(default_factory=list, repr=False, init=False)

    def __post_init__(self):
        self._field_names = [f.name for f in self.__dataclass_fields__.values()
                             if not f.name.startswith('_')]


@dataclass
class SegmentOutput(_DictCompat):
    """Auxiliary segment-level predictions."""

    avg_mut: Optional[torch.Tensor] = None
    segment_id: Optional[torch.Tensor] = None
    avg_kmer_mut: Optional[torch.Tensor] = None

    _field_names: list = field(default_factory=list, repr=False, init=False)

    def __post_init__(self):
        self._field_names = [f.name for f in self.__dataclass_fields__.values()
                             if not f.name.startswith('_')]
