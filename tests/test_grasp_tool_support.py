from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "drogoff",
    "drog",
    "crog",
    "lgd",
    "ggcnnclip",
    "grconvnetclip",
    "etrg",
}


class GraspToolSupportTest(unittest.TestCase):
    def test_embedded_source_assets_are_complete(self):
        assets = ROOT / "assets" / "grasp_tools"
        graspall = assets / "graspall"
        backgrounds = assets / "backgrounds"
        images = sorted(graspall.glob("*.jpg"))
        annotations = sorted(graspall.glob("*.json"))
        self.assertEqual(len(images), 107)
        self.assertEqual(len(annotations), 107)
        self.assertEqual(
            {path.stem for path in images},
            {path.stem for path in annotations},
        )
        self.assertEqual(len(list(backgrounds.glob("*.jpg"))), 42)

    def test_all_requested_profiles_share_dataset_and_schedule(self):
        config_dir = ROOT / "config" / "grasp_tools"
        expected_profiles = MODELS | {
            f"pangu_{model_name}" for model_name in MODELS
        }
        self.assertEqual(
            {path.stem for path in config_dir.glob("*.yaml")},
            expected_profiles,
        )
        for model_name in sorted(MODELS):
            with self.subTest(model=model_name):
                cfg = yaml.safe_load(
                    (config_dir / f"{model_name}.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(cfg["DATA"]["dataset"], "GraspTool")
                self.assertTrue(cfg["DATA"]["dynamic_train_prompts"])
                self.assertEqual(cfg["DATA"]["dynamic_prompt_seed"], 2025)
                self.assertEqual(
                    cfg["DATA"]["root_path"],
                    "./datasets/grasp-tools/aug_graspall_v2",
                )
                self.assertEqual(cfg["MODEL"]["architecture"], model_name)
                self.assertEqual(cfg["TRAIN"]["epochs"], 36)
                self.assertEqual(cfg["TRAIN"]["milestones"], [30])
                self.assertEqual(cfg["TRAIN"]["base_lr"], 0.0001)
                accumulation = cfg["TRAIN"].get(
                    "gradient_accumulation_steps", 1
                )
                self.assertEqual(
                    cfg["TRAIN"]["batch_size"] * accumulation, 32
                )
                self.assertGreater(cfg["TRAIN"]["batch_size_val"], 0)
                self.assertEqual(cfg["TRAIN"]["word_len"], 32)
                self.assertEqual(cfg["TEST"]["test_split"], "test")
                self.assertEqual(
                    cfg["TEST"]["evaluation_protocol"], "crog_legacy"
                )
                self.assertEqual(cfg["TEST"]["grasp_size_activation"], "auto")
                self.assertTrue(cfg["TEST"]["restore_grasp_size_scale"])

                pangu_cfg = yaml.safe_load(
                    (config_dir / f"pangu_{model_name}.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    pangu_cfg["MODEL"]["architecture"],
                    f"pangu_{model_name}",
                )
                self.assertEqual(pangu_cfg["DATA"], cfg["DATA"])
                pangu_train = dict(pangu_cfg["TRAIN"])
                legacy_train = dict(cfg["TRAIN"])
                self.assertEqual(
                    pangu_train.pop("exp_name"),
                    f"pangu_{legacy_train.pop('exp_name')}",
                )
                self.assertEqual(pangu_train, legacy_train)
                self.assertEqual(pangu_cfg["TEST"], cfg["TEST"])

    def test_drogoff_skips_fixed_generator_short_side(self):
        cfg = yaml.safe_load(
            (ROOT / "config" / "grasp_tools" / "drogoff.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(cfg["TRAIN"].get("predict_grasp_short_side", False))
        self.assertNotIn("short_side_loss_weight", cfg["TRAIN"])
        self.assertTrue(cfg["TEST"]["use_offset_at_inference"])

    def test_unstable_film_models_bound_text_conditioning(self):
        ggcnn = (ROOT / "model" / "toolrgs" / "ggcnnclip.py").read_text(
            encoding="utf-8"
        )
        lgd = (ROOT / "model" / "toolrgs" / "lgd.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch.tanh(self.gamma(h))", ggcnn)
        self.assertIn("state = F.normalize(state.float(), dim=-1)", ggcnn)
        self.assertIn("gamma = torch.tanh(gamma)", lgd)
        self.assertIn(
            "text_state = F.normalize(text_state.float(), dim=-1)", lgd
        )
        self.assertIn("eps=1e-4", lgd)
        self.assertIn("if self.contrastive_weight > 0.0:", lgd)
        self.assertIn("_masked_geometry_loss", lgd)
        for model_name in ("ggcnnclip", "lgd"):
            cfg = yaml.safe_load(
                (ROOT / "config" / "grasp_tools" / f"{model_name}.yaml")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(cfg["TRAIN"]["max_norm"], 1.0)
            if model_name == "lgd":
                self.assertEqual(
                    cfg["TRAIN"]["lgd_contrastive_weight"], 0.0
                )

    def test_builder_and_adapter_cover_schema_v21(self):
        builder = (ROOT / "utils" / "data_builder.py").read_text(encoding="utf-8")
        adapter = (ROOT / "utils" / "grasp_tool_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("return GraspToolDataset(", builder)
        self.assertIn('index_path = os.path.join(split_dir, "index.jsonl")', adapter)
        self.assertIn('query = queries[query_index]', adapter)
        self.assertIn(
            'query.get("prompt_cycle", "category_v1") == "category_v1"',
            adapter,
        )
        self.assertIn("include_short=self.with_short_side", adapter)
        self.assertIn('if self.with_short_side:', adapter)
        self.assertIn('getattr(args, "predict_grasp_short_side", False)', builder)
        self.assertIn('grasp_masks["off"]', adapter)

    def test_runner_propagates_epoch_to_dynamic_language_dataset(self):
        runner = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertIn(
            'hasattr(train_loader.dataset, "set_epoch")', runner
        )
        self.assertIn(
            "train_loader.dataset.set_epoch(epoch_log)", runner
        )

    def test_auto_activation_matches_every_model_loss(self):
        clamp_models = (
            "model/crog.py",
            "model/drog.py",
            "model/toolrgs/lgd.py",
            "model/toolrgs/ggcnnclip.py",
            "model/toolrgs/grconvnetclip.py",
            "model/toolrgs/etrg/model.py",
        )
        for relative_path in clamp_models:
            with self.subTest(model=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn('grasp_size_loss_activation = "clamp"', source)
        drogoff = (ROOT / "model" / "drogoff.py").read_text(encoding="utf-8")
        self.assertIn('grasp_size_loss_activation = "sigmoid"', drogoff)

    def test_eight_npu_launcher_preserves_grasp_tool_root(self):
        launcher = (ROOT / "tools" / "train_8npu.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('DATASET_NAME="Grasp-Tools"', launcher)
        self.assertIn("datasets/grasp-tools/aug_graspall_v2", launcher)

    def test_drogoff_uses_true_single_card_batch_32(self):
        cfg = yaml.safe_load(
            (ROOT / "config" / "grasp_tools" / "drogoff.yaml").read_text(
                encoding="utf-8"
            )
        )
        train = cfg["TRAIN"]
        self.assertFalse(train["amp"])
        self.assertFalse(train["find_unused_parameters"])
        self.assertEqual(train["batch_size"], 32)
        self.assertEqual(train["batch_size_val"], 32)
        self.assertNotIn("gradient_accumulation_steps", train)
        runner = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertIn(
            "find_unused_parameters=find_unused_parameters", runner
        )
        layers = (ROOT / "model" / "drog_layers.py").read_text(
            encoding="utf-8"
        )
        projector = layers.split("class MultiTaskProjector", 1)[1].split(
            "class OffsetMultiTaskProjector", 1
        )[0]
        self.assertIn("groups=batch_size,", projector)
        self.assertNotIn("groups=batch_size * branch_count", projector)
        self.assertIn("branch_features[:, branch_index].contiguous()", projector)


if __name__ == "__main__":
    unittest.main()
