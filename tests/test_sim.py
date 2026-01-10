import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from iuca.sim import IterativeWorkingMemorySim  # noqa: E402


class TestSim(unittest.TestCase):
    def test_step_invariants(self) -> None:
        sim = IterativeWorkingMemorySim(N=50, K=8, r=0.5, seed=0)
        overlap, W_next = sim.step()

        self.assertEqual(len(W_next), sim.K)
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)


if __name__ == "__main__":
    unittest.main()
