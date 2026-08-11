from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "crogoff",
    "drog",
    "drogoff",
    "etrg",
    "ggcnnclip",
    "grconvnetclip",
    "graspmamba",
    "lgd",
    "maplegrasp",
)
NAMESPACED_MODELS = (
    "crogoff",
    "ggcnnclip",
    "grconvnetclip",
    "graspmamba",
    "lgd",
    "maplegrasp",
)


class ToolRGSModelMigrationTest(unittest.TestCase):
    def test_all_configs_lock_the_crog_protocol(self):
        for model in MODELS:
            path = ROOT / "config" / "OCID-VLG" / f"{model}.yaml"
            source = path.read_text(encoding="utf-8-sig")
            with self.subTest(model=model):
                self.assertRegex(
                    source,
                    re.compile(
                        rf"^\s*architecture:\s*{model}\s*$",
                        re.MULTILINE,
                    ),
                )
                self.assertRegex(
                    source,
                    re.compile(
                        r"^\s*evaluation_protocol:\s*crog_legacy\s*$",
                        re.MULTILINE,
                    ),
                )
                self.assertRegex(
                    source,
                    re.compile(r"^\s*amp:\s*False\s*$", re.MULTILINE),
                )

    def test_original_crog_stays_the_default_builder(self):
        source = (ROOT / "model" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('getattr(args, "architecture", "crog")', source)
        self.assertIn('"crog": build_crog', source)
        self.assertIn("from .toolrgs import build_toolrgs_model", source)

    def test_migrated_models_are_namespaced(self):
        for model in NAMESPACED_MODELS:
            with self.subTest(model=model):
                self.assertTrue(
                    (ROOT / "model" / "toolrgs" / f"{model}.py").is_file()
                )
        top_level = {path.name for path in (ROOT / "model").glob("*.py")}
        self.assertIn("drogoff.py", top_level)
        self.assertIn("drog.py", top_level)

    def test_etrg_rgb_profile_keeps_official_modules_and_crog_contract(self):
        package = ROOT / "model" / "toolrgs" / "etrg"
        files = (
            "model.py", "bridger.py", "clip.py",
            "fusion.py", "layers.py", "LICENSE",
        )
        for name in files:
            with self.subTest(name=name):
                self.assertTrue((package / name).is_file())
        source = (package / "model.py").read_text(encoding="utf-8")
        for token in (
            "requires_depth = False",
            "supports_offset = False",
            "Bridger_SA_RN_depth",
            "self.resnet18",
            "return detached, target_values, total",
            'ensure_pretrained(cfg.clip_pretrain, "clip-rn50")',
            'ensure_pretrained(local_weight, "resnet18")',
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertNotIn(".cuda(", source)
        config = (ROOT / "config" / "OCID-VLG" / "etrg.yaml").read_text(
            encoding="utf-8"
        )
        self.assertRegex(config, r"(?m)^\s*with_depth:\s*false\s*$")
        self.assertRegex(config, r"(?m)^\s*etrg_input_mode:\s*rgb\s*$")
        self.assertIn("pretrain/resnet18-f37072fd.pth", config)

    def test_crog_evaluation_contract_is_retained(self):
        source = (
            ROOT / "engine" / "crog_engine.py"
        ).read_text(encoding="utf-8")
        expected = (
            "else cv2.INTER_CUBIC",
            "ins_mask_pred = (ins_mask_pred > _segmentation_threshold(args))",
            "return [1, 5]",
            "torch.sigmoid(grasp_qua_mask_preds)",
            "_decode_grasp_size_map(grasp_wid_mask_preds, args)",
            "return calculate_jacquard_index(grasp_predictions, grasp_targets)",
        )
        for token in expected:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_only_offset_models_request_offset_targets(self):
        source = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertIn(
            'needs_offset = bool(getattr(model, "supports_offset", False))',
            source,
        )
        self.assertEqual(source.count("with_grasp_offset=needs_offset"), 1)

    def test_standalone_evaluator_uses_the_same_model_builder_and_npu_path(self):
        source = (ROOT / "test_crog.py").read_text(encoding="utf-8")
        self.assertIn("from model import build_model", source)
        self.assertIn("model, _ = build_model(args)", source)
        self.assertIn(
            '"crog_legacy": "crog_legacy"', source
        )
        self.assertIn(
            '"vcot_official": "vcot_official"', source
        )
        self.assertNotIn(".cuda(", source)
        self.assertNotIn("torch.cuda", source)

    def test_offset_models_print_their_offset_loss(self):
        source = (
            ROOT / "engine" / "crog_engine.py"
        ).read_text(encoding="utf-8")
        self.assertIn("AverageMeter('Loss_off', ':2.4f')", source)
        self.assertIn('off_loss_metter.update(loss_dict["m_off"]', source)
        self.assertIn(
            'getattr(unwrapped_model, "supports_offset", False)',
            source,
        )

    def test_maplegrasp_matches_the_official_two_stage_contract(self):
        source = (
            ROOT / "model" / "toolrgs" / "maplegrasp.py"
        ).read_text(encoding="utf-8")
        expected = (
            "https://github.com/vineet2104/MapleGrasp",
            "c1b1f48e7ff24caaf39daa127d47d9469b93c7a1",
            "self.vis = nn.Sequential",
            "self.vis_mask = nn.Conv2d",
            "self.vis_grasp = nn.Conv2d",
            "self.txt = nn.Linear",
            "torch.sigmoid(mask_out.detach()) > 0.35",
            "F.binary_cross_entropy_with_logits",
            "F.smooth_l1_loss",
        )
        for token in expected:
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertNotIn("maple_stage", source)
        self.assertNotIn(".cuda(", source)

    def test_maplegrasp_stage_configs_are_exclusive_and_linked(self):
        stage1 = (
            ROOT / "config" / "OCID-VLG" / "maplegrasp_stage1.yaml"
        ).read_text(encoding="utf-8")
        stage2 = (
            ROOT / "config" / "OCID-VLG" / "maplegrasp_stage2.yaml"
        ).read_text(encoding="utf-8")
        self.assertRegex(stage1, r"(?m)^\s*stage1:\s*True\s*$")
        self.assertRegex(stage1, r"(?m)^\s*stage2:\s*False\s*$")
        self.assertRegex(stage2, r"(?m)^\s*stage1:\s*False\s*$")
        self.assertRegex(stage2, r"(?m)^\s*stage2:\s*True\s*$")
        self.assertIn(
            "weight: exp/ocid_vlg/maplegrasp_stage1_ocid_vlg_8npu/"
            "best_iou_model.pth",
            stage2,
        )

    def test_maplegrasp_runner_separates_weight_initialization_and_resume(self):
        source = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertIn("def _load_maplegrasp_stage1", source)
        self.assertIn("def _resolve_timestamped_checkpoint", source)
        self.assertIn('"best_iou_model.pth": (', source)
        self.assertIn('"best_iou_epoch_*_IoU_*.pth"', source)
        self.assertIn('"best_epoch_*_IoU_*.pth"', source)
        self.assertIn("model.load_state_dict(state_dict, strict=False)", source)
        self.assertIn('"module.proj.vis_grasp.weight"', source)
        self.assertIn(
            'getattr(model.module, "segmentation_only", False)', source
        )
        self.assertIn("model.load_state_dict(checkpoint['state_dict'])", source)

if __name__ == "__main__":
    unittest.main()
