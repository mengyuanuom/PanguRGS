from pathlib import Path
import importlib.util
import unittest

import numpy as np
import torch
import yaml


from utils.vcot_geometry import grasp_anything_to_quads, resolve_vcot_split
from utils.vcot_eval import (
    calculate_vcot_grasp_success,
    vcot_angle_within_threshold,
    vcot_rotated_iou,
)


ROOT = Path(__file__).resolve().parents[1]
LAYERS_SPEC = importlib.util.spec_from_file_location(
    "crog_layers_for_test", ROOT / "model" / "layers.py"
)
LAYERS_MODULE = importlib.util.module_from_spec(LAYERS_SPEC)
LAYERS_SPEC.loader.exec_module(LAYERS_MODULE)
MultiTaskProjector = LAYERS_MODULE.MultiTaskProjector


class VCoTDrogoffSupportTest(unittest.TestCase):
    def test_split_aliases_match_official_files(self):
        self.assertEqual(resolve_vcot_split("train"), "train.csv")
        self.assertEqual(resolve_vcot_split("seen"), "test_seen.csv")
        self.assertEqual(resolve_vcot_split("unseen"), "test_unseen.csv")

    def test_grasp_anything_rows_convert_to_xy_quads(self):
        quads, scores = grasp_anything_to_quads(
            np.array([[0.9, 50.0, 40.0, 20.0, 10.0, 0.0]], dtype=np.float32)
        )
        self.assertEqual(quads.shape, (1, 4, 2))
        self.assertAlmostEqual(float(scores[0]), 0.9, places=6)
        self.assertTrue(np.allclose(quads.mean(axis=1)[0], [50.0, 40.0]))

    def test_official_metric_is_single_prediction_and_inclusive(self):
        base = [50.0, 50.0, 40.0, 20.0, 0.0]
        boundary = [74.0, 50.0, 40.0, 20.0, 30.0]
        self.assertTrue(vcot_angle_within_threshold(0.0, 30.0))
        self.assertGreaterEqual(vcot_rotated_iou(base, base), 1.0 - 1e-6)
        self.assertEqual(calculate_vcot_grasp_success(base, [base]), 1)
        self.assertEqual(calculate_vcot_grasp_success(None, [boundary]), 0)

    def test_drogoff_profile_uses_vcot_paths_and_schedule(self):
        path = ROOT / "config" / "vcot" / "drogoff.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["DATA"]["dataset"], "vcot")
        self.assertEqual(
            cfg["DATA"]["root_path"], "./datasets/graspanything-vcot"
        )
        self.assertEqual(cfg["DATA"]["train_split"], "train")
        self.assertEqual(cfg["DATA"]["val_split"], "unseen")
        self.assertEqual(cfg["MODEL"]["architecture"], "drogoff")
        self.assertEqual(cfg["TRAIN"]["batch_size"], 32)
        self.assertEqual(cfg["TRAIN"]["base_lr"], 0.0001)
        self.assertEqual(cfg["TRAIN"]["epochs"], 36)
        self.assertEqual(cfg["TRAIN"]["milestones"], [30])
        self.assertEqual(cfg["TRAIN"]["val_start_epoch"], 11)
        self.assertEqual(cfg["TRAIN"]["save_epochs"], [5, 10])
        self.assertEqual(cfg["TEST"]["test_split"], "unseen")
        self.assertEqual(cfg["TEST"]["evaluation_protocol"], "vcot_official")
        self.assertEqual(cfg["TEST"]["grasp_topk"], [1])
        self.assertEqual(cfg["TEST"]["grasp_size_activation"], "auto")
        self.assertTrue(cfg["TRAIN"]["predict_grasp_short_side"])
        self.assertEqual(cfg["TRAIN"]["short_side_loss_weight"], 1.0)
        self.assertTrue(cfg["TEST"]["restore_grasp_size_scale"])
        self.assertEqual(cfg["DATA"]["grasp_size_factor"], 300)

    def test_vcot_crog_predicts_long_and_short_sides(self):
        path = ROOT / "config" / "vcot" / "crog.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["MODEL"]["architecture"], "crog")
        self.assertEqual(cfg["DATA"]["grasp_size_factor"], 300)
        self.assertTrue(cfg["TRAIN"]["predict_grasp_short_side"])
        self.assertEqual(cfg["TRAIN"]["short_side_loss_weight"], 1.0)

        model = (ROOT / "model" / "crog.py").read_text(encoding="utf-8")
        self.assertIn("predicts_grasp_short_side", model)
        self.assertIn('loss_dict["m_short"]', model)

        image_features = torch.randn(2, 4, 3, 3)
        text_features = torch.randn(2, 8)
        legacy = MultiTaskProjector(word_dim=8, in_dim=2, kernel_size=3)
        vcot = MultiTaskProjector(
            word_dim=8,
            in_dim=2,
            kernel_size=3,
            predict_short_side=True,
        )
        self.assertEqual(len(legacy(image_features, text_features)), 5)
        outputs = vcot(image_features, text_features)
        self.assertEqual(len(outputs), 6)
        self.assertTrue(all(output.shape == (2, 1, 12, 12) for output in outputs))

    def test_train_test_and_launcher_route_vcot(self):
        trainer = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        evaluator = (ROOT / "test_crog.py").read_text(encoding="utf-8")
        builder = (ROOT / "utils" / "data_builder.py").read_text(encoding="utf-8")
        engine = (ROOT / "engine" / "crog_engine.py").read_text(encoding="utf-8")
        launcher = (ROOT / "tools" / "train_8npu.sh").read_text(encoding="utf-8")
        self.assertEqual(trainer.count("build_referring_grasp_dataset("), 2)
        self.assertIn('"vcot_official"', trainer)
        self.assertIn("DistributedEvalSampler(val_data)", trainer)
        self.assertNotIn("DistributedSampler(val_data", trainer)
        self.assertIn("build_referring_grasp_dataset(", evaluator)
        self.assertIn("return VCoTDataset(", builder)
        adapter = (ROOT / "utils" / "vcot_dataset.py").read_text(encoding="utf-8")
        self.assertNotIn('"depth":', adapter)
        self.assertIn("calculate_vcot_grasp_success", engine)
        self.assertIn("return 0.5 if _is_vcot_official(args) else 0.35", engine)
        self.assertIn("topk != [1]", engine)
        self.assertIn("return topk", engine)
        self.assertIn("datasets/graspanything-vcot", launcher)
        self.assertIn('TRAIN_OPTS+=(DATA.split_root "${SPLIT_ROOT}")', launcher)


    def test_short_side_decode_and_resample_restore_scale(self):
        dataset = (ROOT / "utils" / "dataset.py").read_text(encoding="utf-8")
        model = (ROOT / "model" / "drogoff.py").read_text(encoding="utf-8")
        engine = (ROOT / "engine" / "crog_engine.py").read_text(encoding="utf-8")
        grasp_eval = (ROOT / "utils" / "grasp_eval.py").read_text(encoding="utf-8")
        offset_eval = (ROOT / "utils" / "offset_eval.py").read_text(encoding="utf-8")
        self.assertIn("'short': short_out", dataset)
        self.assertIn("predicts_grasp_short_side", model)
        self.assertIn("short_side_loss", model)
        self.assertIn("grasp_short_mask=None", grasp_eval)
        self.assertIn("short_side=None", offset_eval)
        self.assertIn("restore_grasp_size_scale", engine)

    def test_grasp_size_losses_use_the_inference_value_range(self):
        model = (ROOT / "model" / "drogoff.py").read_text(encoding="utf-8")
        self.assertIn(
            "torch.sigmoid(width), grasp_wid_mask",
            model,
        )
        self.assertIn(
            "torch.sigmoid(short_side), grasp_short_mask",
            model,
        )


if __name__ == "__main__":
    unittest.main()