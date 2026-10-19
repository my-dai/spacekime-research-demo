import tempfile
import unittest
from pathlib import Path

from spacekime_paths import (
    candidate_mat_paths,
    portable_manifest_path,
    resolve_mat_path,
)


class SpacekimePathTests(unittest.TestCase):
    def test_explicit_existing_path_wins(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mat") as handle:
            resolved = resolve_mat_path(handle.name)
            self.assertEqual(resolved, Path(handle.name).resolve())

    def test_candidates_are_deduplicated(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mat") as handle:
            candidates = candidate_mat_paths(handle.name)
            resolved = [str(path.resolve()).lower() for path in candidates]
            self.assertEqual(len(resolved), len(set(resolved)))

    def test_manifest_paths_are_relative_or_redacted_to_a_basename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            inside = root / "results" / "run"
            outside = Path(tmp) / "private" / "run"
            self.assertEqual(
                portable_manifest_path(inside, root=root), "results/run"
            )
            self.assertEqual(portable_manifest_path(outside, root=root), "run")


if __name__ == "__main__":
    unittest.main()
