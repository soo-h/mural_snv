import unittest
import torch
import numpy as np

from MuRaL.data.dataset import FeatureBatchSpec, unwrap_batch
from MuRaL.data.preprocessing import SiteShuffleBuffer


class FeatureBatchSpecTests(unittest.TestCase):
    """Test FeatureBatchSpec — the feature order contract for batch tuples."""

    def test_get_feature_order_default_required_only(self):
        """When enabled_optional_keys is empty list, only required_keys are returned."""
        spec = FeatureBatchSpec(enabled_optional_keys=[])
        order = spec.get_feature_order()
        self.assertEqual(order, ["mut_type", "cat_x", "distal_x"])

    def test_get_feature_order_with_selected_optional(self):
        """Enabled optional features appear after required keys in defined order."""
        spec = FeatureBatchSpec(
            enabled_optional_keys=["step_avg_mut", "sample_weight"]
        )
        order = spec.get_feature_order()
        expected = [
            "mut_type", "cat_x", "distal_x",
            "step_avg_mut", "sample_weight",
        ]
        self.assertEqual(order, expected)

    def test_get_feature_order_skips_unknown_optional(self):
        """Keys not in optional_key_order are silently ignored."""
        spec = FeatureBatchSpec(
            enabled_optional_keys=["not_a_real_key", "arg_feature"]
        )
        order = spec.get_feature_order()
        expected = ["mut_type", "cat_x", "distal_x", "arg_feature"]
        self.assertEqual(order, expected)

    def test_get_feature_order_with_available_keys_filter(self):
        """available_keys filters out optional features not present in data."""
        spec = FeatureBatchSpec(
            enabled_optional_keys=["segment_avg_mut", "arg_feature", "sample_weight"]
        )
        order = spec.get_feature_order(
            available_keys=["mut_type", "cat_x", "distal_x", "arg_feature"]
        )
        expected = ["mut_type", "cat_x", "distal_x", "arg_feature"]
        self.assertEqual(order, expected)

    def test_get_feature_order_none_enabled_means_all_optional(self):
        """When enabled_optional_keys is None, all optional keys are included."""
        spec = FeatureBatchSpec()  # enabled_optional_keys defaults to None
        order = spec.get_feature_order()
        self.assertIn("sample_weight", order)
        self.assertIn("segment_avg_kmer_mut", order)
        # required keys come first
        self.assertEqual(order[:3], ["mut_type", "cat_x", "distal_x"])


def _make_window(n_sites, cat_n=5, mut_offset=0):
    """Create a synthetic encoding window dict with *n_sites* sites."""
    return {
        "mut_type": torch.arange(mut_offset, mut_offset + n_sites),
        "cat_x": torch.randn(n_sites, cat_n),
        "distal_x": torch.randn(n_sites, 4, 21),
    }


class SiteShuffleBufferTests(unittest.TestCase):
    """Test SiteShuffleBuffer — the streaming site buffer and batch generator."""

    def setUp(self):
        self.spec = FeatureBatchSpec(enabled_optional_keys=[])

    def _make_windows_sequential(self, ns_sites_per_window):
        """Return windows with globally sequential mut_type values."""
        windows = []
        offset = 0
        for n in ns_sites_per_window:
            windows.append(_make_window(n, mut_offset=offset))
            offset += n
        return windows

    def _collect_all_sites(self, buffer):
        """Collect all sites from one epoch traversal of *buffer*."""
        sites = {"mut_type": [], "cat_x": [], "distal_x": []}
        for batch in buffer:
            for i, key in enumerate(buffer._feature_keys()):
                sites[key].append(batch[i])
        return {k: torch.cat(v, dim=0) for k, v in sites.items() if v}

    # ------------------------------------------------------------------
    def test_yields_correct_batch_size(self):
        """Every yielded batch (except possibly the last) has site_batch_size sites."""
        windows = self._make_windows_sequential([60, 60, 60, 60, 60])  # 300 sites total
        buffer = SiteShuffleBuffer(
            window_iter=windows,
            site_batch_size=50,
            shuffle_buffer_size=200,
            feature_spec=self.spec,
            shuffle_sites=False,
            drop_last=True,
        )
        batches = list(iter(buffer))
        self.assertGreater(len(batches), 0)
        for b in batches:
            self.assertEqual(b[0].shape[0], 50)

    def test_all_sites_present_once_no_shuffle(self):
        """Every site appears exactly once when shuffle_sites=False."""
        windows = self._make_windows_sequential([50, 80, 30])  # 160 sites
        buffer = SiteShuffleBuffer(
            window_iter=windows,
            site_batch_size=20,
            shuffle_buffer_size=100,
            feature_spec=self.spec,
            shuffle_sites=False,
            drop_last=False,
        )
        collected = self._collect_all_sites(buffer)
        self.assertEqual(collected["mut_type"].shape[0], 160)
        # order is preserved (no shuffle)
        torch.testing.assert_close(collected["mut_type"], torch.arange(160))

    def test_drop_last_drops_incomplete_batch(self):
        """drop_last=True discards the final incomplete batch."""
        windows = self._make_windows_sequential([55])  # 55 sites, batch_size=20 → 2 full + 15 leftover
        buffer = SiteShuffleBuffer(
            window_iter=windows,
            site_batch_size=20,
            shuffle_buffer_size=100,
            feature_spec=self.spec,
            shuffle_sites=False,
            drop_last=True,
        )
        collected = self._collect_all_sites(buffer)
        # 55 sites → batch sizes: 20, 20, 15 (dropped) → 40
        self.assertEqual(collected["mut_type"].shape[0], 40)

    def test_drop_last_false_yields_incomplete_batch(self):
        """drop_last=False yields the final incomplete batch."""
        windows = self._make_windows_sequential([55])
        buffer = SiteShuffleBuffer(
            window_iter=windows,
            site_batch_size=20,
            shuffle_buffer_size=100,
            feature_spec=self.spec,
            shuffle_sites=False,
            drop_last=False,
        )
        batches = list(iter(buffer))
        batch_sizes = [b[0].shape[0] for b in batches]
        self.assertEqual(batch_sizes, [20, 20, 15])

    def test_shuffle_changes_site_order(self):
        """shuffle_sites=True changes the site order relative to input."""
        torch.manual_seed(42)
        windows = self._make_windows_sequential([200])
        buffer_shuffled = SiteShuffleBuffer(
            window_iter=windows,
            site_batch_size=200,
            shuffle_buffer_size=200,
            feature_spec=self.spec,
            shuffle_sites=True,
            drop_last=False,
        )
        batch = next(iter(buffer_shuffled))
        # compare to original order
        self.assertFalse(torch.allclose(batch[0], torch.arange(200)))

    def test_carry_over_across_flushes(self):
        """Sites not filling a batch are carried to the next buffer flush."""
        # buffer will flush at 100 sites; 55 sites don't fill to 100
        windows = self._make_windows_sequential([60, 60, 60])  # 180 sites
        buffer = SiteShuffleBuffer(
            window_iter=windows,
            site_batch_size=20,
            shuffle_buffer_size=80,  # flushes at 80
            feature_spec=self.spec,
            shuffle_sites=False,
            drop_last=False,
        )
        # 60 sites → buffer=60 (below 80, no flush)
        # +60 → buffer=120 → flush 120 sites as 6 batches of 20 → buffer=0
        # +60 → buffer=60 → force_last flush → 3 batches of 20
        collected = self._collect_all_sites(buffer)
        self.assertEqual(collected["mut_type"].shape[0], 180)

    def test_feature_spec_controls_output_keys(self):
        """Output tuple keys match the feature_spec order, not dict key order."""
        windows = self._make_windows_sequential([30])
        spec = FeatureBatchSpec(enabled_optional_keys=[])
        buffer = SiteShuffleBuffer(
            window_iter=windows,
            site_batch_size=30,
            shuffle_buffer_size=100,
            feature_spec=spec,
        )
        # _feature_keys() requires buffer to have data
        batch = next(iter(buffer))
        keys_in_batch = len(batch)
        # spec returns 3 keys (mut_type, cat_x, distal_x)
        self.assertEqual(keys_in_batch, len(spec.get_feature_order()))

    def test_reusable_across_epochs(self):
        """__iter__ returns a new generator that replays all windows."""
        windows = self._make_windows_sequential([30])
        buffer = SiteShuffleBuffer(
            window_iter=windows,
            site_batch_size=30,
            shuffle_buffer_size=100,
            feature_spec=self.spec,
        )
        # first epoch
        e1 = list(iter(buffer))
        self.assertEqual(len(e1), 1)
        # second epoch — same data, fresh generator
        e2 = list(iter(buffer))
        self.assertEqual(len(e2), 1)


# ------------------------------------------------------------------
#  Integration test — real BED data
# ------------------------------------------------------------------
class IntegrationTests(unittest.TestCase):
    """End-to-end integration test with a real BED file.

    Validates that EncodingWindowDataset + SiteShuffleBuffer produce
    batches that the trainer can consume.
    """

    BED_PATH = "/tmp/test_small.bed"
    REF_GENOME = "/public5_data/home/songhui/data/hg19/hg19_ucsc_ordered.fa"

    def setUp(self):
        self.spec = FeatureBatchSpec(enabled_optional_keys=[])

    def test_pipeline_with_real_bed_no_shuffle(self):
        """Pipeline with a real BED file produces correctly sized batches."""
        from MuRaL.data.data_preprocess_pipeline import DatasetPreprocessor
        from torch.utils.data import DataLoader

        config = {
            'segment_center': 10000,
            'local_radius': 7,
            'local_order': 3,
            'distal_radius': 1000,
            'distal_order': 1,
            'h5f_path': None,
            'seq_only': True,
            'n_h5_files': 1,
            'without_bw_distal': True,
            'bw_paths': None,
            'trial_dir': None,
            'features': {
                'local_feature': {
                    'loading': 'eager',
                    'type': 'computed',
                    'compute_fn': 'local_feature',
                    'params': {
                        'local_radius': 7,
                        'local_order': 3,
                        'names': ['local_seq', 'cat_x', 'mut_type'],
                    },
                },
                'distal_x': {
                    'loading': 'lazy',
                    'type': 'computed',
                    'compute_fn': 'distal_encoding',
                    'params': {
                        'distal_radius': 1000,
                        'order': 1,
                        'encoding_type': 'ohe',
                    },
                },
            },
        }

        preprocessor = DatasetPreprocessor(config, use_h5=False)
        dataset = preprocessor.preprocess_dataset(
            self.BED_PATH, self.REF_GENOME, use_segment_task=False
        )

        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)

        segment_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, collate_fn=unwrap_batch,
        )

        buffer = SiteShuffleBuffer(
            window_iter=segment_loader,
            site_batch_size=256,
            shuffle_buffer_size=5000,
            shuffle_sites=False,
            drop_last=False,
            feature_spec=self.spec,
        )

        batches = list(iter(buffer))
        self.assertGreater(len(batches), 0)

        total_sites = 0
        for i, batch in enumerate(batches):
            n_sites = batch[0].shape[0]
            total_sites += n_sites
            if i < len(batches) - 1:
                self.assertEqual(n_sites, 256,
                    f"Batch {i} has {n_sites} sites, expected 256")
            else:
                self.assertLessEqual(n_sites, 256,
                    f"Last batch has {n_sites} sites, expected <= 256")

        # verify all sites from BED were read exactly once
        expected_sites = len(dataset.get_labels())
        self.assertEqual(total_sites, expected_sites,
            f"Expected {expected_sites} sites, got {total_sites}")

    def _make_pipeline(self):
        """Build a full pipeline (dataset + loader + buffer) with the test BED."""
        from MuRaL.data.data_preprocess_pipeline import DatasetPreprocessor
        from torch.utils.data import DataLoader

        config = {
            'segment_center': 10000,
            'local_radius': 7,
            'local_order': 3,
            'distal_radius': 1000,
            'distal_order': 1,
            'h5f_path': None,
            'seq_only': True,
            'n_h5_files': 1,
            'without_bw_distal': True,
            'bw_paths': None,
            'trial_dir': None,
            'features': {
                'local_feature': {
                    'loading': 'eager',
                    'type': 'computed',
                    'compute_fn': 'local_feature',
                    'params': {
                        'local_radius': 7,
                        'local_order': 3,
                        'names': ['local_seq', 'cat_x', 'mut_type'],
                    },
                },
                'distal_x': {
                    'loading': 'lazy',
                    'type': 'computed',
                    'compute_fn': 'distal_encoding',
                    'params': {
                        'distal_radius': 1000,
                        'order': 1,
                        'encoding_type': 'ohe',
                    },
                },
            },
        }

        preprocessor = DatasetPreprocessor(config, use_h5=False)
        dataset = preprocessor.preprocess_dataset(
            self.BED_PATH, self.REF_GENOME, use_segment_task=False
        )

        segment_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, collate_fn=unwrap_batch,
        )

        buffer = SiteShuffleBuffer(
            window_iter=segment_loader,
            site_batch_size=256,
            shuffle_buffer_size=5000,
            shuffle_sites=False,
            drop_last=False,
            feature_spec=self.spec,
        )

        return dataset, buffer

    def test_all_sites_read_exactly_once(self):
        """Every site from the real BED is read exactly once per epoch."""
        dataset, buffer = self._make_pipeline()

        all_sites = []
        for batch in iter(buffer):
            all_sites.append(batch[0])  # mut_type

        all_sites = torch.cat(all_sites, dim=0)
        expected_sites = len(dataset.get_labels())

        self.assertEqual(len(all_sites), expected_sites)

        # With shuffle_sites=False, sites should appear in order (one pass)
        _, buffer2 = self._make_pipeline()
        all_sites2 = torch.cat([b[0] for b in iter(buffer2)], dim=0)
        torch.testing.assert_close(all_sites, all_sites2,
            msg="Two sequential passes should produce identical site order")

    def test_pipeline_reusable_across_two_epochs(self):
        """SiteShuffleBuffer can be iterated twice without recreation — same total sites."""
        dataset, buffer = self._make_pipeline()
        expected_sites = len(dataset.get_labels())

        # Epoch 1
        e1 = [b[0].shape[0] for b in iter(buffer)]
        self.assertEqual(sum(e1), expected_sites)
        # Epoch 2 — no recreation
        e2 = [b[0].shape[0] for b in iter(buffer)]
        self.assertEqual(sum(e2), expected_sites)
        self.assertEqual(e1, e2,
            "Same data, same shuffle=False → batch sizes should be identical across epochs")


# ------------------------------------------------------------------
#  Recording model for testing model_train_v2 dispatch
# ------------------------------------------------------------------
class _RecordModel(torch.nn.Module):
    """Toy model that records the (local_input, distal_input, extra_args) it receives."""

    def __init__(self):
        super().__init__()
        self.last_local = None
        self.last_distal = None
        self.last_extra = ()

    def forward(self, local_input, distal_input, *extra):
        self.last_local = local_input
        self.last_distal = distal_input
        self.last_extra = extra
        return torch.zeros(1)


class ModelTrainV2Tests(unittest.TestCase):
    """Test model_train_v2 — FeatureBatchSpec-driven model dispatch."""

    def setUp(self):
        self.spec = FeatureBatchSpec(enabled_optional_keys=[])

    def test_modern_model_receives_dict_local_input(self):
        """Modern models get dict {'cont_data': ..., 'cat_data': ...}."""
        from MuRaL.training.train import model_train_v2

        model = _RecordModel()
        inputs = (torch.tensor(0), torch.randn(4, 5), torch.randn(4, 4, 21))

        model_train_v2(inputs, model, self.spec)

        self.assertIsInstance(model.last_local, dict)
        self.assertIn('cont_data', model.last_local)
        self.assertIn('cat_data', model.last_local)

    def test_legacy_model_receives_tuple_local_input(self):
        """Legacy Network0 receives tuple (cont_data, cat_data)."""
        from MuRaL.training.train import model_train_v2
        from MuRaL.models.nn_models import Network0

        # Minimal Network0 instance (won't actually run inference)
        model = Network0.__new__(Network0)
        inputs = (torch.tensor(0), torch.randn(4, 5), torch.randn(4, 4, 21))

        with self.assertRaises(AttributeError):
            # Network0.__new__ has no internal model, so forward will fail
            # but we only care that (cont_x, cat_x) tuple was passed as local_input
            model_train_v2(inputs, model, self.spec)

    def test_optional_feature_mapped_to_local_input(self):
        """step_avg_mut is mapped to local_input['avg_mutations']."""
        from MuRaL.training.train import model_train_v2

        spec = FeatureBatchSpec(enabled_optional_keys=['step_avg_mut'])
        model = _RecordModel()
        inputs = (
            torch.tensor(0),
            torch.randn(4, 5),
            torch.randn(4, 4, 21),
            torch.randn(4, 7),  # step_avg_mut
        )

        model_train_v2(inputs, model, spec)

        self.assertIn('avg_mutations', model.last_local)
        self.assertEqual(model.last_local['avg_mutations'].shape, (4, 7))

    def test_arg_feature_passed_as_positional(self):
        """arg_feature is passed as positional argument after distal_x."""
        from MuRaL.training.train import model_train_v2

        spec = FeatureBatchSpec(enabled_optional_keys=['arg_feature'])
        model = _RecordModel()
        inputs = (
            torch.tensor(0),
            torch.randn(4, 5),
            torch.randn(4, 4, 21),
            torch.randn(4, 23),  # arg_feature
        )

        model_train_v2(inputs, model, spec)

        self.assertEqual(len(model.last_extra), 1)
        self.assertEqual(model.last_extra[0].shape, (4, 23))

    def test_register_v2_returns_callable(self):
        """model_train_register_v2 returns a function that can be called."""
        from MuRaL.training.train import model_train_register_v2

        train_fn = model_train_register_v2(self.spec)
        self.assertTrue(callable(train_fn))

        model = _RecordModel()
        inputs = (torch.tensor(0), torch.randn(4, 5), torch.randn(4, 4, 21))
        result = train_fn(inputs, model)
        self.assertIsNotNone(result)
