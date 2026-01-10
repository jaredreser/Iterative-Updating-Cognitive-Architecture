import sys
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from iuca.selection import gumbel_topk_straight_through  # noqa: E402


class TestSelection(unittest.TestCase):
    def test_topk_mask_count(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4, 10)
        hard, soft = gumbel_topk_straight_through(logits, k=3, tau=1.0)

        self.assertEqual(hard.shape, logits.shape)
        self.assertTrue(torch.all(hard.sum(dim=-1) == 3))
        self.assertTrue(torch.allclose(soft.sum(dim=-1), torch.ones(4), atol=1e-5))

    def test_zero_k(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 5)
        hard, soft = gumbel_topk_straight_through(logits, k=0, tau=1.0)

        self.assertTrue(torch.all(hard == 0))
        self.assertTrue(torch.allclose(soft.sum(dim=-1), torch.ones(2), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
