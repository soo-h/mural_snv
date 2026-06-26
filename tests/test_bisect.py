import unittest
import torch

from MuRaL.data.dataset import FeatureBatchSpec, unwrap_batch, dict_to_tuple_collate
from MuRaL.data.preprocessing import SiteShuffleBuffer, generate_data_batches
from torch.utils.data import DataLoader

# Minimal config matching the integration test
from MuRaL.data.data_preprocess_pipeline import DatasetPreprocessor


class BisectTests(unittest.TestCase):
    """Compare old and new pipeline output side-by-side."""

    BED_PATH = "/tmp/test_small.bed"
    REF_GENOME = "/public5_data/home/songhui/data/hg19/hg19_ucsc_ordered.fa"

    def _make_config(self):
        return {
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
                    'loading': 'eager', 'type': 'computed', 'compute_fn': 'local_feature',
                    'params': {'local_radius': 7, 'local_order': 3,
                               'names': ['local_seq', 'cat_x', 'mut_type']},
                },
                'distal_x': {
                    'loading': 'lazy', 'type': 'computed', 'compute_fn': 'distal_encoding',
                    'params': {'distal_radius': 1000, 'order': 1, 'encoding_type': 'ohe'},
                },
            },
        }

    def test_old_vs_new_batch_identical(self):
        """Old and new pipelines produce identical batch contents (no shuffle)."""
        config = self._make_config()
        spec = FeatureBatchSpec(enabled_optional_keys=[])

        # --- OLD pipeline ---
        preprocessor_old = DatasetPreprocessor(config, use_h5=False)
        dataset_old = preprocessor_old.preprocess_dataset(
            self.BED_PATH, self.REF_GENOME, use_segment_task=False
        )
        seg_loader_old = DataLoader(
            dataset_old, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, collate_fn=dict_to_tuple_collate,
        )
        old_data = generate_data_batches(
            seg_loader_old, batch_segment=3, batch_size=32, shuffle=False,
        )

        # --- NEW pipeline ---
        preprocessor_new = DatasetPreprocessor(config, use_h5=False)
        dataset_new = preprocessor_new.preprocess_dataset(
            self.BED_PATH, self.REF_GENOME, use_segment_task=False
        )
        seg_loader_new = DataLoader(
            dataset_new, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, collate_fn=unwrap_batch,
        )
        buffer = SiteShuffleBuffer(
            window_iter=seg_loader_new,
            site_batch_size=32,
            shuffle_buffer_size=1000,
            shuffle_sites=False,
            feature_spec=spec,
        )

        # Convert both to list of batches and compare
        new_batches = list(iter(buffer))
        old_batches = list(old_data)

        # Same number of sites total
        new_total = sum(b[0].shape[0] for b in new_batches)
        old_total = sum(b[0].shape[0] for b in old_batches)
        self.assertEqual(new_total, old_total,
            f"New: {new_total} sites, Old: {old_total} sites")

        # Compare first batch element-by-element
        merged_new = torch.cat([b[0] for b in new_batches], dim=0)
        merged_old = torch.cat([b[0] for b in old_batches], dim=0)
        torch.testing.assert_close(merged_new, merged_old,
            msg="mut_type values differ between old and new pipeline")

        # Compare cat_x
        merged_new_cat = torch.cat([b[1] for b in new_batches], dim=0)
        merged_old_cat = torch.cat([b[1] for b in old_batches], dim=0)
        torch.testing.assert_close(merged_new_cat, merged_old_cat,
            msg="cat_x values differ between old and new pipeline")

        # Compare distal_x
        merged_new_distal = torch.cat([b[2] for b in new_batches], dim=0)
        merged_old_distal = torch.cat([b[2] for b in old_batches], dim=0)
        torch.testing.assert_close(merged_new_distal, merged_old_distal,
            msg="distal_x values differ between old and new pipeline")

        print(f"  Old vs New: {new_total} sites, all feature values match")

    def test_get_inputs_labels_old_vs_new(self):
        """get_inputs_labels_v2 produces same result as get_inputs_labels for same batch."""
        from MuRaL.training.train import get_inputs_labels_v2, get_inputs_labels

        config = self._make_config()
        spec = FeatureBatchSpec(enabled_optional_keys=[])

        preprocessor = DatasetPreprocessor(config, use_h5=False)
        dataset = preprocessor.preprocess_dataset(
            self.BED_PATH, self.REF_GENOME, use_segment_task=False
        )
        seg_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False, collate_fn=unwrap_batch,
        )
        buffer = SiteShuffleBuffer(
            window_iter=seg_loader, site_batch_size=32,
            shuffle_buffer_size=1000, shuffle_sites=False, feature_spec=spec,
        )

        for batch in iter(buffer):
            # Old path
            labels_old, inputs_old, sw_old = get_inputs_labels(batch)
            # New path
            labels_new, inputs_new, sw_new = get_inputs_labels_v2(batch, spec)

            # Labels
            self.assertEqual(labels_old['label'].shape, labels_new['label'].shape)
            torch.testing.assert_close(labels_old['label'], labels_new['label'],
                msg="label differs")

            # Inputs
            self.assertEqual(len(inputs_old), len(inputs_new),
                f"Inputs length differs: old={len(inputs_old)}, new={len(inputs_new)}")
            for i, (io, inew) in enumerate(zip(inputs_old, inputs_new)):
                if isinstance(io, torch.Tensor):
                    torch.testing.assert_close(io, inew,
                        msg=f"inputs[{i}] differs. Old shape={io.shape}, new shape={inew.shape}")
                else:
                    self.assertEqual(io, inew, msg=f"inputs[{i}] scalar differs")

            # Sample weight
            if sw_old is not None and sw_new is not None:
                torch.testing.assert_close(sw_old, sw_new)
            else:
                self.assertEqual(sw_old, sw_new)

        print("  get_inputs_labels: old == new for all batches")

    def test_old_vs_new_ska_local_style(self):
        """Old and new produce same model call for SKA_local-like batch."""
        from MuRaL.training.train import (get_inputs_labels_v2, get_inputs_labels,
                                           model_train_v2, model_train_register_v2,
                                           model_train_register)

        # Build a batch that looks like SKA_local output:
        # (mut_type, cat_x, distal_x, step_avg_mut, segment_avg_kmer_mut, arg_feature, sample_weight)
        bs = 16
        batch = (
            torch.arange(bs),                          # mut_type
            torch.randn(bs, 13),                       # cat_x
            torch.randn(bs, 4, 2001),                  # distal_x
            torch.randn(bs, 7),                        # step_avg_mut
            torch.randn(bs, 14, 3, 4),                 # segment_avg_kmer_mut
            torch.randn(bs, 23),                       # arg_feature
            torch.randn(bs),                            # sample_weight
        )

        spec = FeatureBatchSpec(
            enabled_optional_keys=['step_avg_mut', 'segment_avg_kmer_mut', 'arg_feature', 'sample_weight']
        )

        # Old path
        labels_old, inputs_old, sw_old = get_inputs_labels(batch, 'SKA_local')
        # New path
        labels_new, inputs_new, sw_new = get_inputs_labels_v2(batch, spec, 'SKA_local')

        # Compare labels
        self.assertEqual(set(labels_old.keys()), set(labels_new.keys()),
            f"Label keys differ: old={set(labels_old.keys())}, new={set(labels_new.keys())}")
        for k in labels_old:
            torch.testing.assert_close(labels_old[k], labels_new[k],
                msg=f"labels['{k}'] differs")

        # Compare inputs
        self.assertEqual(len(inputs_old), len(inputs_new),
            f"Input length: old={len(inputs_old)}, new={len(inputs_new)}")
        for i, (io, inew) in enumerate(zip(inputs_old, inputs_new)):
            if isinstance(io, torch.Tensor):
                torch.testing.assert_close(io, inew,
                    msg=f"inputs[{i}] differs. Old shape={io.shape}, new shape={inew.shape}")

        # Compare sample_weight
        if sw_old is not None:
            torch.testing.assert_close(sw_old, sw_new)
        else:
            self.assertEqual(sw_old, sw_new)

        # Compare model calls
        class RecordModel(torch.nn.Module):
            def forward(self, local_input, distal_x, *extra):
                self.last_local = local_input
                self.last_distal = distal_x
                self.last_extra = extra
                return torch.zeros(bs)
            def __init__(self):
                super().__init__()

        # Old model call
        old_model = RecordModel()
        old_train_fn = model_train_register('SKA_local')  # old register
        old_model_call = old_train_fn(inputs_old, old_model)

        # New model call
        new_model = RecordModel()
        new_train_fn = model_train_register_v2(spec)
        new_model_call = new_train_fn(inputs_new, new_model)

        # Compare local_input dicts
        self.assertEqual(set(old_model.last_local.keys()), set(new_model.last_local.keys()),
            f"local_input keys differ: old={set(old_model.last_local.keys())}, "
            f"new={set(new_model.last_local.keys())}")
        for k in old_model.last_local:
            torch.testing.assert_close(old_model.last_local[k], new_model.last_local[k],
                msg=f"local_input['{k}'] differs")

        # Compare positional args
        self.assertEqual(len(old_model.last_extra), len(new_model.last_extra),
            f"extra positional args count: old={len(old_model.last_extra)}, new={len(new_model.last_extra)}")
        for i, (oe, ne) in enumerate(zip(old_model.last_extra, new_model.last_extra)):
            torch.testing.assert_close(oe, ne,
                msg=f"extra[{i}] differs")

        print("  SKA_local: old == new for labels, inputs, model call")
