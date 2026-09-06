"""Actual CPU tensor regression for calibration counts independent of H/score rows."""
import unittest
import torch
from prismaquant.tessera_campaign import _collect_activations


class CaptureCountTests(unittest.TestCase):
    def test_dense_capture_counts_all_rows_without_hessian_or_retained_rows(self):
        net = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False)).eval()
        batches = [torch.arange(12, dtype=torch.float32).reshape(3, 4),
                   torch.arange(12, 24, dtype=torch.float32).reshape(3, 4)]
        for cap in (0, 2):
            with self.subTest(max_rows=cap):
                rows, hess, counts, maxima = _collect_activations(
                    net, ['0'], batches, max_rows=cap, device='cpu', want_hessian=False)
                self.assertEqual(counts, {'0': 6})
                self.assertEqual(hess, {})
                self.assertEqual(maxima, {'0': 23.0})
                self.assertEqual(None if rows['0'] is None else rows['0'].shape[0], None if cap == 0 else cap)

    def test_hessian_mode_keeps_full_gram_and_same_row_count(self):
        net = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False)).eval()
        batches = [torch.arange(24, dtype=torch.float32).reshape(6, 4)]
        rows, hess, counts, maxima = _collect_activations(
            net, ['0'], batches, max_rows=0, device='cpu', want_hessian=True)
        self.assertEqual(counts, {'0': 6})
        self.assertIsNone(rows['0'])
        torch.testing.assert_close(hess['0'], batches[0].T @ batches[0])


if __name__ == '__main__':
    unittest.main()
