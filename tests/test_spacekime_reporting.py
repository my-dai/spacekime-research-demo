import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from spacekime_reporting import build_run_manifest, write_run_manifest


class SpacekimeReportingTests(unittest.TestCase):
    def test_manifest_hides_local_path_and_has_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mat_path = Path(tmp) / "tiny.mat"
            savemat(mat_path, {"bigim": np.ones((2, 2, 1, 2), dtype=np.complex64)})
            manifest = build_run_manifest(
                "test",
                mat_path,
                parameters={"sigma": 1.0},
                results={"value": float("nan")},
            )
            dataset = manifest["dataset"]
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(dataset["basename"], "tiny.mat")
            self.assertNotIn("local_path", dataset)
            self.assertEqual(len(dataset["fingerprint"]["value"]), 64)

            output = Path(tmp) / "manifest.json"
            write_run_manifest(output, manifest)
            parsed = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(parsed["results"]["value"])


if __name__ == "__main__":
    unittest.main()
