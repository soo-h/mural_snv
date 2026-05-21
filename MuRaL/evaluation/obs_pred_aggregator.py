import numpy as np

class ObsPredAggregator:
    """Aggregate observed counts and predicted values by grouping key."""

    def __init__(self, n_class):
        self.n_class = n_class
        self.obs = {}
        self.pred = {}
        self.site_count = {}

    def add_obs(self, key, mut_type, count=1):
        if key not in self.obs:
            self.obs[key] = np.zeros(self.n_class)
            self.site_count[key] = 0
        self.obs[key][mut_type] += count
        self.site_count[key] += 1

    def add_pred(self, key, probs):
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

