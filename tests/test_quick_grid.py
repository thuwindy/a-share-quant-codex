from __future__ import annotations

import unittest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ashare_quant.analysis.quick_grid import parse_float_grid, parse_int_grid


class QuickGridParseTest(unittest.TestCase):
    def test_parse_int_grid_deduplicates_and_preserves_order(self) -> None:
        values = parse_int_grid("20, 30, 20,40", fallback=[12])
        self.assertEqual(values, [20, 30, 40])

    def test_parse_int_grid_fallback_when_empty(self) -> None:
        values = parse_int_grid("", fallback=[8, 12])
        self.assertEqual(values, [8, 12])

    def test_parse_float_grid_deduplicates_and_preserves_order(self) -> None:
        values = parse_float_grid("0.03,0.05,0.03", fallback=[0.01])
        self.assertEqual(values, [0.03, 0.05])

    def test_parse_float_grid_fallback_when_empty(self) -> None:
        values = parse_float_grid("   ", fallback=[0.02, 0.03])
        self.assertEqual(values, [0.02, 0.03])


if __name__ == "__main__":
    unittest.main()
