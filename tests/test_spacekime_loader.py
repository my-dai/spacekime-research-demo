import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spacekime_loader import load_bigim


class SpacekimeLoaderTests(unittest.TestCase):
    @patch("spacekime_loader.sio.loadmat")
    def test_load_bigim_requests_only_bigim_variable(self, loadmat) -> None:
        loadmat.return_value = {
            "bigim": np.ones((2, 3, 1, 4), dtype=np.complex128)
        }
        with tempfile.NamedTemporaryFile(suffix=".mat") as handle:
            result = load_bigim(handle.name)

        self.assertEqual(result.dtype, np.complex64)
        self.assertTrue(result.flags.c_contiguous)
        loadmat.assert_called_once_with(Path(handle.name), variable_names=["bigim"])


if __name__ == "__main__":
    unittest.main()
