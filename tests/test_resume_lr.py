from pathlib import Path
import unittest

from utils.lr_scheduler import calculate_multistep_lrs


class ResumeLearningRateTest(unittest.TestCase):
    def test_milestone_15_is_active_after_epoch_15(self):
        lrs = calculate_multistep_lrs(
            base_lrs=[1e-4],
            milestones=[15],
            gamma=0.1,
            completed_epochs=15,
        )
        self.assertAlmostEqual(lrs[0], 1e-5)

    def test_resume_path_rebuilds_instead_of_loading_old_scheduler(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "train_crog.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "scheduler.load_state_dict(checkpoint['scheduler'])",
            source,
        )
        self.assertIn("scheduler = rebuild_multistep_scheduler(", source)
        self.assertIn("completed_epochs=args.start_epoch", source)


if __name__ == "__main__":
    unittest.main()
