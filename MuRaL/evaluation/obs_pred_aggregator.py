import numpy as np

class ObsPredAggregator:
    """Aggregate observed counts and predicted values by grouping key.

    Supports multi-allelic sites via the is_new_site flag: when a single
    genomic position carries multiple mutation labels (e.g., recurrent
    mode), set is_new_site=False for duplicate occurrences so that:
      - obs: all labels are accumulated (each allele counts)
      - pred: only the first occurrence contributes (one prob per site)
      - site_count: counts unique sites (denominator for rate averages)
    """

    def __init__(self, n_class):
        self.n_class = n_class
        self.obs = {}
        self.pred = {}
        self.site_count = {}

    def add_obs(self, key, mut_type, count=1, is_new_site=True):
        if key not in self.obs:
            self.obs[key] = np.zeros(self.n_class)
            self.site_count[key] = 0
        self.obs[key][mut_type] += count
        if is_new_site:
            self.site_count[key] += 1

    def add_pred(self, key, probs, is_new_site=True):
        if not is_new_site:
            return
        if key not in self.pred:
            self.pred[key] = np.zeros(self.n_class)
        for i in range(self.n_class):
            self.pred[key][i] += probs[i]

class KmerMutSaver(ObsPredAggregator):
    """Aggregate by kmer sequence."""
    pass


class RegionMutSaver(ObsPredAggregator):
    """Aggregate by (chrom, window_end) tuple."""
    pass

