"""Lightweight regression tests. After installing requirements, run:
    python -m unittest test_core.py -v
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import (
    analyze_fold,
    draw_rose,
    figure_bytes,
    prepare_measurements,
    rose_histogram,
)


ROOT = Path(__file__).resolve().parent


class StructuralGeologyTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.read_csv(ROOT / "sample_field_data.csv")
        self.data = prepare_measurements(self.raw, "RHR Strike / Dip")

    def test_sample_contents(self):
        self.assertEqual(len(self.data), 40)
        self.assertEqual(int(self.data.Is_Plane.sum()), 38)
        self.assertEqual(int(self.data.Is_Line.sum()), 2)
        self.assertEqual(int(self.data.Structure_Type.str.contains("Bedding").sum()), 28)

    def test_fold_axis_known_synthetic_geometry(self):
        beds = self.data[self.data.Structure_Type.str.contains("Bedding")]
        result = analyze_fold(beds)
        self.assertLess(abs(result.fold_trend - 115), 8)
        self.assertLess(abs(result.fold_plunge - 10), 8)
        self.assertGreater(result.eigenvalues[1], result.eigenvalues[2])

    def test_dip_direction_mode_gives_same_rhr(self):
        inp = self.raw.copy()
        inp["Dip_Direction"] = (pd.to_numeric(inp["Strike"]) + 90) % 360
        converted = prepare_measurements(inp, "Dip Direction / Dip")
        np.testing.assert_allclose(
            self.data.Strike_RHR.dropna().to_numpy(),
            converted.Strike_RHR.dropna().to_numpy(),
        )

    def test_bad_angles_rejected(self):
        for column, value in [("Strike", 361), ("Dip", 91), ("Plunge", -5)]:
            inp = self.raw.copy()
            row = 39 if column == "Plunge" else 0
            inp.loc[row, column] = value
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    prepare_measurements(inp, "RHR Strike / Dip")

    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            prepare_measurements(self.raw.drop(columns="Dip"), "RHR Strike / Dip")

    def test_rose_counts_and_binary_exports(self):
        angles = self.data.loc[self.data.Is_Plane, "Strike_RHR"].to_numpy()
        axial, _ = rose_histogram(angles, 10, bidirectional=True)
        directional, _ = rose_histogram(angles, 15, bidirectional=False)
        self.assertEqual(int(axial.sum()), 2 * len(angles))
        self.assertEqual(int(directional.sum()), len(angles))
        fig = draw_rose(angles, 10, True, "Test rose")
        self.assertTrue(figure_bytes(fig, "png").startswith(b"\x89PNG"))
        self.assertIn(b"<svg", figure_bytes(fig, "svg")[:1200])
        import matplotlib.pyplot as plt
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
