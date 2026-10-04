"""Experiment cache discovery must never silently pick an arbitrary run."""

import tempfile
import unittest
from pathlib import Path

from company_reviews.benchmark import resolve_cache_path


class CacheDiscoveryTest(unittest.TestCase):
    def test_unique_cache_is_discovered_and_explicit_path_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "mpnet_cpu.json"
            first.write_text("{}")
            self.assertEqual(
                resolve_cache_path(None, root, "mpnet*.json", "--mpnet-manifest"), first
            )
            second = root / "mpnet_mps.json"
            second.write_text("{}")
            with self.assertRaisesRegex(ValueError, "--mpnet-manifest"):
                resolve_cache_path(None, root, "mpnet*.json", "--mpnet-manifest")
            self.assertEqual(
                resolve_cache_path(second, root, "mpnet*.json", "--mpnet-manifest"),
                second,
            )

    def test_missing_cache_gives_actionable_error(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(FileNotFoundError, "notebook"),
        ):
            resolve_cache_path(None, Path(directory), "mpnet*.json", "--mpnet-manifest")
