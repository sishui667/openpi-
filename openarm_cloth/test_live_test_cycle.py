"""Retention boundaries; fixtures live only in an isolated temporary directory."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from live_test_cycle import cleanup_candidates, managed_run, finish_test_cycle


class RetentionFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="openarm_retention_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()

    def make_run(self, name="20261016_220000_000001", family="live_bimanual",
                 finished=True, schema="openarm_live_test_v1"):
        run = self.outputs / family / name
        dataset = run / "dataset"
        dataset.mkdir(parents=True)
        (dataset / "manifest.json").write_text(
            json.dumps({"schema": schema, "finished": finished}), encoding="utf-8")
        return run

    def symlink(self, target, link, is_directory=False):
        try:
            os.symlink(target, link, target_is_directory=is_directory)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Creating symlinks unavailable: {error}")


class ManagedRunTests(RetentionFixture):
    def test_accepts_exact_direct_run_under_either_live_family(self):
        for family in ("live_human", "live_bimanual"):
            with self.subTest(family=family):
                run = self.make_run(family=family)
                self.assertEqual(managed_run(run, self.outputs), run.resolve())

    def test_rejects_outputs_root_and_family_roots(self):
        run = self.make_run()
        for path in (self.outputs, run.parent):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    managed_run(path, self.outputs)

    def test_rejects_other_output_families(self):
        run = self.make_run(family="formal_episodes")
        with self.assertRaises(ValueError):
            managed_run(run, self.outputs)

    def test_rejects_non_timestamp_run_name(self):
        run = self.make_run(name="latest")
        with self.assertRaises(ValueError):
            managed_run(run, self.outputs)

    def test_rejects_nested_descendant(self):
        run = self.make_run()
        with self.assertRaises(ValueError):
            managed_run(run / "dataset", self.outputs)

    def test_rejects_sibling_with_matching_path_prefix(self):
        outside = self.root / "outputs_backup" / "live_bimanual" / "20261016_220000_000001"
        outside.mkdir(parents=True)
        with self.assertRaises(ValueError):
            managed_run(outside, self.outputs)

    def test_rejects_symlink_run_even_when_target_is_another_managed_run(self):
        target = self.make_run(name="20261016_210000_000001")
        link = target.parent / "20261016_220000_000001"
        self.symlink(target, link, is_directory=True)
        with self.assertRaises(ValueError):
            managed_run(link, self.outputs)

    def test_rejects_symlink_inside_dataset(self):
        run = self.make_run()
        target = self.root / "outside_payload.npz"
        target.write_bytes(b"not a real archive")
        self.symlink(target, run / "dataset" / "chunk_00000.npz")
        with self.assertRaises(ValueError):
            managed_run(run, self.outputs)


class CleanupCandidateTests(RetentionFixture):
    def test_lists_only_older_recognized_completed_runs_across_live_families(self):
        current = self.make_run(name="20261016_230000_000001")
        old_dual = self.make_run(name="20261016_220000_000001")
        old_single = self.make_run(name="20261016_210000_000001", family="live_human")
        self.make_run(name="20261016_235900_000001")
        candidates = cleanup_candidates(current, self.outputs)
        self.assertEqual(set(candidates), {old_dual.resolve(), old_single.resolve()})
        self.assertNotIn(current.resolve(), candidates)

    def test_unfinished_recording_is_never_a_cleanup_candidate(self):
        current = self.make_run(name="20261016_230000_000001")
        self.make_run(finished=False)
        self.assertEqual(cleanup_candidates(current, self.outputs), [])

    def test_unknown_schema_is_never_a_cleanup_candidate(self):
        current = self.make_run(name="20261016_230000_000001")
        self.make_run(schema="unrelated_formal_dataset")
        self.assertEqual(cleanup_candidates(current, self.outputs), [])

    def test_missing_or_malformed_manifest_is_never_a_cleanup_candidate(self):
        current = self.make_run(name="20261016_230000_000001")
        missing = self.outputs / "live_human" / "20261016_210000_000001"
        (missing / "dataset").mkdir(parents=True)
        malformed = self.make_run()
        (malformed / "dataset" / "manifest.json").write_text("broken json", encoding="utf-8")
        self.assertEqual(cleanup_candidates(current, self.outputs), [])

    def test_unknown_top_level_file_prevents_entire_run_cleanup(self):
        current = self.make_run(name="20261016_230000_000001")
        old = self.make_run()
        (old / "important_notes.md").write_text("user-owned", encoding="utf-8")
        self.assertEqual(cleanup_candidates(current, self.outputs), [])

    def test_unknown_dataset_file_prevents_entire_run_cleanup(self):
        current = self.make_run(name="20261016_230000_000001")
        old = self.make_run()
        (old / "dataset" / "weights.pt").write_bytes(b"user-owned weights")
        self.assertEqual(cleanup_candidates(current, self.outputs), [])

    def test_non_timestamp_directory_is_not_touched(self):
        current = self.make_run(name="20261016_230000_000001")
        self.make_run(name="saved_best_demonstration")
        self.assertEqual(cleanup_candidates(current, self.outputs), [])

    def test_symlink_tree_is_not_a_cleanup_candidate(self):
        current = self.make_run(name="20261016_230000_000001")
        old = self.make_run()
        outside = self.root / "outside_payload.npz"
        outside.write_bytes(b"user-owned")
        self.symlink(outside, old / "dataset" / "chunk_00000.npz")
        self.assertEqual(cleanup_candidates(current, self.outputs), [])

    def test_formal_episode_directory_is_not_a_cleanup_candidate(self):
        current = self.make_run(name="20261016_230000_000001")
        self.make_run(family="formal_episodes")
        self.assertEqual(cleanup_candidates(current, self.outputs), [])


class FinishCycleTests(RetentionFixture):
    def ready_run(self):
        current = self.make_run(name='20261016_230000_000001')
        (current / 'dataset' / 'manifest.json').write_text(json.dumps(dict(
            schema='openarm_live_test_v1', finished=True, run_error=None)), encoding='utf-8')
        (current / 'summary.json').write_text('{"run_error": null}', encoding='utf-8')
        return current

    def test_metadata_is_archived_before_recycling_and_latest_is_not_recycled(self):
        current = self.ready_run()
        old = self.make_run()
        recycled = []
        def recycle(path):
            history = self.outputs / 'test_cycle_history' / f'{old.parent.name}_{old.name}.json'
            self.assertTrue(history.is_file())
            self.assertFalse(json.loads(history.read_text())['recycled'])
            recycled.append(path)
        with patch('validate_live_test_dataset.validate', return_value=dict(valid=True, recording_complete=True)), \
                patch('live_test_cycle.analyze_useful', return_value=dict(eligible=True, useful_rows=20)):
            result = finish_test_cycle(current, self.outputs, recycle=recycle)
        self.assertTrue(result['eligible'])
        self.assertEqual(recycled, [old.resolve()])
        self.assertTrue(current.exists())
        saved = json.loads((current / 'optimization_report.json').read_text())
        self.assertTrue(saved['eligible'])
        self.assertFalse(saved['training_performed'])

    def test_blank_new_test_preserves_old_data(self):
        current = self.ready_run()
        old = self.make_run()
        recycled = []
        with patch('validate_live_test_dataset.validate', return_value=dict(valid=True, recording_complete=True)), \
                patch('live_test_cycle.analyze_useful', return_value=dict(eligible=False, useful_rows=0)):
            result = finish_test_cycle(current, self.outputs, recycle=recycled.append)
        self.assertFalse(result['eligible'])
        self.assertEqual(recycled, [])
        self.assertTrue(old.exists())

    def test_failed_new_test_preserves_old_data_even_if_writer_finished(self):
        current = self.ready_run()
        old = self.make_run()
        (current / 'summary.json').write_text('{"run_error": "camera failed"}', encoding='utf-8')
        recycled = []
        result = finish_test_cycle(current, self.outputs, recycle=recycled.append)
        self.assertFalse(result['eligible'])
        self.assertEqual(recycled, [])
        self.assertTrue(old.exists())

    def test_dry_run_writes_nothing_and_recycles_nothing(self):
        current = self.ready_run()
        self.make_run()
        recycled = []
        with patch('validate_live_test_dataset.validate', return_value=dict(valid=True, recording_complete=True)), \
                patch('live_test_cycle.analyze_useful', return_value=dict(eligible=True, useful_rows=20)):
            result = finish_test_cycle(current, self.outputs, apply=False, recycle=recycled.append)
        self.assertTrue(result['eligible'])
        self.assertEqual(recycled, [])
        self.assertFalse((self.outputs / 'test_cycle_history').exists())


if __name__ == "__main__":
    unittest.main()
