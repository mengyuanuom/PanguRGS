import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model_package = types.ModuleType("model")
model_package.__path__ = [str(ROOT / "model")]
sys.modules.setdefault("model", model_package)
toolrgs_package = types.ModuleType("model.toolrgs")
toolrgs_package.__path__ = [str(ROOT / "model" / "toolrgs")]
sys.modules.setdefault("model.toolrgs", toolrgs_package)
_load_module(
    "model.toolrgs.crog_clip", ROOT / "model" / "toolrgs" / "crog_clip.py"
)
_load_module(
    "model.toolrgs.crog_layers", ROOT / "model" / "toolrgs" / "crog_layers.py"
)
MAPLE_MODULE = _load_module(
    "model.toolrgs.maplegrasp", ROOT / "model" / "toolrgs" / "maplegrasp.py"
)
MultiTaskProjectorPP = MAPLE_MODULE.MultiTaskProjectorPP


class VCoTMapleGraspTest(unittest.TestCase):
    def test_optional_short_side_preserves_legacy_output_shape(self):
        image_features = torch.randn(2, 4, 3, 3)
        text_features = torch.randn(2, 8)
        legacy = MultiTaskProjectorPP(
            word_dim=8, in_dim=2, kernel_size=3, stage2=True
        )
        vcot = MultiTaskProjectorPP(
            word_dim=8,
            in_dim=2,
            kernel_size=3,
            stage2=True,
            predict_short_side=True,
        )

        legacy_outputs = legacy(image_features, text_features)
        vcot_outputs = vcot(image_features, text_features)
        self.assertEqual(len(legacy_outputs), 5)
        self.assertEqual(len(vcot_outputs), 6)
        self.assertEqual(legacy.vis_grasp.out_channels, 8)
        self.assertEqual(vcot.vis_grasp.out_channels, 10)
        self.assertTrue(
            all(output.shape == (2, 1, 12, 12) for output in vcot_outputs)
        )

    def test_vcot_stage_profiles_are_linked_and_use_dual_sizes(self):
        stage1_path = ROOT / "config" / "vcot" / "maplegrasp_stage1.yaml"
        stage2_path = ROOT / "config" / "vcot" / "maplegrasp_stage2.yaml"
        stage1 = yaml.safe_load(stage1_path.read_text(encoding="utf-8"))
        stage2 = yaml.safe_load(stage2_path.read_text(encoding="utf-8"))

        self.assertTrue(stage1["TRAIN"]["stage1"])
        self.assertFalse(stage1["TRAIN"]["stage2"])
        self.assertFalse(stage2["TRAIN"]["stage1"])
        self.assertTrue(stage2["TRAIN"]["stage2"])
        self.assertEqual(stage2["DATA"]["dataset"], "vcot")
        self.assertEqual(stage2["DATA"]["grasp_size_factor"], 300)
        self.assertTrue(stage2["TRAIN"]["predict_grasp_short_side"])
        self.assertTrue(stage2["TRAIN"]["align_grasp_size_loss"])
        self.assertEqual(stage2["TRAIN"]["short_side_loss_weight"], 1.0)
        self.assertEqual(
            stage2["TRAIN"]["weight"],
            "exp/vcot/maplegrasp_stage1_vcot_seen_8npu/"
            "best_iou_model.pth",
        )
        self.assertEqual(
            stage2["TEST"]["evaluation_protocol"], "vcot_official"
        )
        self.assertEqual(stage2["TEST"]["grasp_size_activation"], "auto")
        self.assertEqual(stage2["TEST"]["test_split"], "unseen")

    def test_vcot_size_loss_and_metadata_are_matched(self):
        source = (
            ROOT / "model" / "toolrgs" / "maplegrasp.py"
        ).read_text(encoding="utf-8")
        self.assertIn("predicts_grasp_short_side", source)
        self.assertIn('"sigmoid" if self.align_grasp_size_loss else None', source)
        self.assertIn("torch.sigmoid(outputs[index])", source)
        self.assertIn('loss_dict["m_short"]', source)


if __name__ == "__main__":
    unittest.main()
