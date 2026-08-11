from pathlib import Path
import re
import unittest
from utils.config import (
    CfgNode,
    merge_cfg_from_list,
    resolve_grasp_size_activation,
)


ROOT = Path(__file__).resolve().parents[1]


class OfficialCROGNPUConfigTest(unittest.TestCase):
    def test_none_config_placeholder_accepts_checkpoint_path(self):
        checkpoint_path = "exp/run/best_jindex_model.pth"
        cfg = CfgNode({"resume": None})
        merged = merge_cfg_from_list(cfg, ["TRAIN.resume", checkpoint_path])
        self.assertEqual(merged.resume, checkpoint_path)

    def test_auto_grasp_size_activation_matches_checkpoint_training(self):
        self.assertEqual(resolve_grasp_size_activation("auto", {}), "clamp")
        self.assertEqual(
            resolve_grasp_size_activation(
                "auto", {"grasp_size_activation": "sigmoid"}
            ),
            "sigmoid",
        )
        self.assertEqual(
            resolve_grasp_size_activation(
                "clamp", {"grasp_size_activation": "sigmoid"}
            ),
            "clamp",
        )

    def test_official_training_hyperparameters_are_preserved(self):
        path = ROOT / "config" / "OCID-VLG" / "crog_multiple_r50.yaml"
        source = path.read_text(encoding="utf-8")
        expected_lines = (
            r"^\s*epochs:\s*36\s*$",
            r"^\s*milestones:\s*\[30\]\s*$",
            r"^\s*batch_size:\s*32\b",
            r"^\s*batch_size_val:\s*32\b",
            r"^\s*base_lr:\s*0\.0001\b",
            r"^\s*lr_multi:\s*0\.1\b",
            r"^\s*sync_bn:\s*False\b",
            r"^\s*amp:\s*False\b",
            r"^\s*pin_memory:\s*False\s*$",
            r"^\s*with_depth:\s*False\s*$",
            r"^\s*resume:\s*$",
            r"^\s*save_freq:\s*0\s*$",
            r"^\s*val_freq:\s*1\b",
            r"^\s*val_start_epoch:\s*11\s*$",
            r"^\s*save_epochs:\s*\[5,\s*10\]\s*$",
            r"^\s*dist_backend:\s*['\"]hccl['\"]\s*$",
            r"^\s*dist_url:\s*env://\s*$",
        )
        for pattern in expected_lines:
            with self.subTest(pattern=pattern):
                self.assertRegex(source, re.compile(pattern, re.MULTILINE))

    def test_yaml_training_profiles_match_selected_schedule(self):
        config_dir = ROOT / "config"
        long_schedule = {
            "OCID-VLG/drogoff.yaml",
            "OCID-VLG/drog.yaml",
            "OCID-VLG/crog_multiple_r50.yaml",
            "OCID-VLG/lgd.yaml",
            "OCID-VLG/grconvnetclip.yaml",
            "OCID-VLG/ggcnnclip.yaml",
            "OCID-VLG/etrg.yaml",
            "vcot/drogoff.yaml",
            "vcot/crog.yaml",
        }
        for path in sorted(config_dir.rglob("*.yaml")):
            source = path.read_text(encoding="utf-8")
            relative_config = path.relative_to(config_dir).as_posix()
            canonical_config = relative_config.replace("/pangu_", "/", 1)
            is_long_run = (
                canonical_config in long_schedule
                or canonical_config.startswith("grasp_tools/")
            )
            expected_epochs = 36 if is_long_run else 24
            expected_milestone = 30 if is_long_run else 20
            expected_batch = (
                256 if canonical_config == "vcot/crog.yaml" else 32
            )
            with self.subTest(config=relative_config):
                self.assertRegex(
                    source,
                    rf"(?m)^\s*batch_size:\s*{expected_batch}\b",
                )
                self.assertRegex(
                    source,
                    rf"(?m)^\s*batch_size_val:\s*{expected_batch}\b",
                )
                self.assertRegex(
                    source, rf"(?m)^\s*epochs:\s*{expected_epochs}\s*$"
                )
                self.assertRegex(
                    source,
                    rf"(?m)^\s*milestones:\s*\[{expected_milestone}\]\s*$",
                )
                self.assertRegex(source, r"(?m)^\s*base_lr:\s*0\.0001\b")
                self.assertRegex(source, r"(?m)^\s*save_freq:\s*0\s*$")
                self.assertRegex(source, r"(?m)^\s*val_freq:\s*1\s*$")
                self.assertRegex(source, r"(?m)^\s*val_start_epoch:\s*11\s*$")
                self.assertRegex(source, r"(?m)^\s*save_epochs:\s*\[5,\s*10\]\s*$")

    def test_training_path_has_no_cuda_or_nccl_calls(self):
        paths = (
            ROOT / "train_crog.py",
            ROOT / "test_crog.py",
            ROOT / "engine" / "crog_engine.py",
            ROOT / "utils" / "misc.py",
            ROOT / "utils" / "npu.py",
        )
        forbidden = (".cuda(", "torch.cuda", "'nccl'", '"nccl"')
        for path in paths:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                for token in forbidden:
                    self.assertNotIn(token, source)

    def test_single_npu_evaluation_loads_ddp_checkpoints(self):
        source = (ROOT / "test_crog.py").read_text(encoding="utf-8")
        self.assertIn("set_device(args.npu)", source)
        self.assertIn('map_location="cpu"', source)
        self.assertIn('key.removeprefix("module.")', source)
        self.assertIn("split=test_split", source)
        self.assertNotIn("torch.nn.DataParallel", source)

    def test_multi_npu_evaluation_shards_and_reduces_metrics(self):
        evaluator = (ROOT / "test_crog.py").read_text(encoding="utf-8")
        engine = (ROOT / "engine" / "crog_engine.py").read_text(encoding="utf-8")
        self.assertIn('backend="hccl"', evaluator)
        self.assertIn("range(args.rank, len(full_test_data), args.world_size)", evaluator)
        self.assertIn("dist.all_reduce(stats, op=dist.ReduceOp.SUM)", engine)
        self.assertIn("disable=rank != 0", engine)


    def test_sync_batchnorm_is_disabled_for_all_npu_configs(self):
        config_root = ROOT / "config"
        for path in config_root.rglob("*.yaml"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotRegex(
                    source,
                    re.compile(r"^\s*sync_bn:\s*True\b", re.MULTILINE),
                )

        trainer = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertNotIn("convert_sync_batchnorm", trainer)
        self.assertIn("args.sync_bn = False", trainer)

    def test_clip_download_uses_the_official_hashed_url(self):
        source = (
            ROOT / "tools" / "download_clip_rn50.py"
        ).read_text(encoding="utf-8")
        digest = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
        self.assertIn("https://openaipublic.azureedge.net/clip/models/", source)
        self.assertGreaterEqual(source.count(digest), 2)

    def test_crog_dataloader_does_not_require_depth(self):
        dataset_source = (ROOT / "utils" / "dataset.py").read_text(encoding="utf-8")
        train_source = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        crog_dataset_source = dataset_source.split(
            "class OCIDVLGDataset", 1
        )[1].split("class OCIDGraspDataset", 1)[0]
        self.assertRegex(crog_dataset_source, r"with_depth\s*=\s*False")
        self.assertIn('if "depth" in batch[0]:', crog_dataset_source)
        self.assertNotIn(
            '"depth": torch.stack([torch.from_numpy(x["depth"]) for x in batch])',
            crog_dataset_source,
        )
        builder_source = (
            ROOT / "utils" / "data_builder.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'with_depth=bool(getattr(args, "with_depth", False))',
            builder_source,
        )
        self.assertEqual(
            train_source.count("build_referring_grasp_dataset("), 2
        )

    def test_launcher_requires_one_positional_config(self):
        source = (
            ROOT / "tools" / "train_8npu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ "$#" -ne 1 ]]', source)
        self.assertIn('CONFIG="$1"', source)
        self.assertNotIn('CONFIG="${CONFIG:-', source)

    def test_launcher_reads_amp_only_from_yaml(self):
        source = (
            ROOT / "tools" / "train_8npu.sh"
        ).read_text(encoding="utf-8")
        self.assertNotRegex(source, r'(?m)^\s*(?:export\s+)?AMP=')
        self.assertNotIn('TRAIN.amp', source)

    def test_launcher_and_worker_bind_all_eight_npus(self):
        launcher = (
            ROOT / "tools" / "train_8npu.sh"
        ).read_text(encoding="utf-8")
        trainer = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertIn("0,1,2,3,4,5,6,7", launcher)
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', launcher)
        self.assertIn('--nproc_per_node="${NPROC_PER_NODE}"', launcher)
        self.assertIn('os.environ.get("LOCAL_RANK", 0)', trainer)
        self.assertIn("set_device(local_rank)", trainer)
        self.assertIn('backend="hccl"', trainer)
        self.assertIn("DistributedDataParallel(", trainer)

    def test_fp32_path_does_not_construct_an_npu_grad_scaler(self):
        runtime = (ROOT / "utils" / "npu.py").read_text(encoding="utf-8")
        launcher = (
            ROOT / "tools" / "train_8npu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("class NoOpGradScaler:", runtime)
        self.assertIn("if not enabled:\n        return NoOpGradScaler()", runtime)
        self.assertNotIn("TRAIN.amp", launcher)

    def test_adam_avoids_the_unstable_npu_foreach_kernel(self):
        trainer = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        config = (
            ROOT / "config" / "OCID-VLG" / "etrg.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'getattr(args, "optimizer_foreach", False)', trainer
        )
        self.assertIn("foreach=optimizer_foreach", trainer)
        self.assertIn(
            'param_group["foreach"] = optimizer_foreach', trainer
        )
        self.assertRegex(
            config, r"(?m)^\s*optimizer_foreach:\s*False\s*$"
        )

    def test_crog_evaluator_always_removes_the_mask_channel(self):
        engine = (
            ROOT / "engine" / "crog_engine.py"
        ).read_text(encoding="utf-8")
        extraction = "ins_mask_preds[idx].squeeze().cpu().numpy()"
        self.assertEqual(engine.count(extraction), 3)
        self.assertNotIn(
            "ins_mask_preds[idx].cpu().numpy()", engine
        )
    def test_source_has_no_removed_numpy_scalar_aliases(self):
        removed = (
            re.compile(r"\bnp\.int0\b"),
            re.compile(r"\bnp\.float\b"),
            re.compile(r"\bnp\.int\b"),
            re.compile(r"\bnp\.bool\b"),
        )
        for path in ROOT.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(ROOT)):
                for pattern in removed:
                    self.assertIsNone(pattern.search(source))

    def test_each_training_run_uses_a_timestamped_output_directory(self):
        trainer = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        launchers = (
            ROOT / "tools" / "train_8npu.sh",
            ROOT / "tools" / "train_toolrgs_model_8npu.sh",
        )
        self.assertIn(
            'os.environ.get("CROG_RUN_TIMESTAMP", "").strip()', trainer
        )
        self.assertIn(
            'args.exp_name = f"{base_exp_name}_{run_timestamp}"', trainer
        )
        self.assertIn(
            '%Y%m%d_%H%M%S_%f', trainer
        )
        for launcher in launchers:
            with self.subTest(launcher=launcher.name):
                source = launcher.read_text(encoding="utf-8")
                self.assertIn("CROG_RUN_TIMESTAMP", source)
                self.assertIn("%Y%m%d_%H%M%S_%3N", source)

    def test_validation_interval_and_epoch_checkpoint_names_are_explicit(self):
        source = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        self.assertIn("epoch_log >= val_start_epoch", source)
        self.assertIn(
            "(epoch_log - val_start_epoch) % val_freq == 0", source
        )
        self.assertIn("save_recovery = epoch_log in save_epochs", source)
        self.assertIn("'evaluated': do_eval", source)
        self.assertIn("'last_eval_epoch': last_eval_epoch", source)
        self.assertIn('f"epoch_{epoch_log:03d}_model.pth"', source)
        self.assertNotIn("_replace_epoch_alias", source)
        self.assertIn('"latest_model.pth"', source)
        self.assertIn("'best_j1': best_j1", source)
        self.assertIn("'best_j5': best_j5", source)
        self.assertIn("save_best_j1", source)
        self.assertIn("save_best_j5", source)
        self.assertIn("grasp_sr_topk_limit", source)
        self.assertIn("'grasp_sr_topk': grasp_sr_topk", source)
        self.assertIn("_sync_grasp_sr_topk(", source)
        self.assertIn("_restore_grasp_sr_topk(", source)
        self.assertIn(
            'iou_prefix = "best" if segmentation_only else "best_iou"',
            source,
        )
        self.assertIn('metric_prefix = "best_j1"', source)
        self.assertIn('"best_j5",', source)
        self.assertIn('f"IoU_{100.0 * float(iou):.2f}"', source)
        self.assertIn(
            'f"J1_{100.0 * j1:.2f}_J5_{100.0 * j5:.2f}"', source
        )
        self.assertIn(
            'output_dir.glob(f"{prefix}_epoch_*.pth")', source
        )
        self.assertIn('checkpoint.get("best_j1", legacy_best_j1)', source)
        self.assertIn('checkpoint.get("best_j5", -1.0)', source)
        self.assertIn('"best_jindex_model.pth": (', source)
        self.assertIn('"best_j1_model.pth": (', source)
        self.assertIn('"best_j5_model.pth": (', source)
        self.assertIn(
            "for metric_pattern in metric_patterns", source
        )
        self.assertIn("os.remove(temporary_checkpoint)", source)

    def test_vcot_keeps_five_ranked_grasp_sr_checkpoints(self):
        trainer = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        config = (ROOT / "config" / "vcot" / "drogoff.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _select_grasp_sr_topk", trainer)
        self.assertIn('output_dir.glob("top_graspsr_epoch_*.pth")', trainer)
        self.assertRegex(
            config,
            r"(?m)^\s*grasp_sr_topk:\s*5\s*(?:#.*)?$",
        )


    def test_drog_family_uses_crog_scorer_after_model_postprocess(self):
        trainer = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        evaluator = (ROOT / "test_crog.py").read_text(encoding="utf-8")
        engine = (ROOT / "engine" / "crog_engine.py").read_text(encoding="utf-8")
        inference = engine.split("def inference_with_grasp", 1)[1]

        self.assertIn("from model import build_model", trainer)
        self.assertIn("from model import build_model", evaluator)
        self.assertIn("model, _ = build_model(args)", evaluator)
        self.assertIn("pred[5]", inference)
        self.assertIn("_apply_model_offset(", inference)
        self.assertLess(
            inference.index("_apply_model_offset("),
            inference.index("_calculate_grasp_success("),
        )
        for token in (
            "_inverse_interpolation(args)",
            "align_corners=True",
            "ins_mask_pred = (ins_mask_pred > _segmentation_threshold(args))",
            "num_grasps = _evaluation_topk(args)",
            "detect_grasps(",
            "_calculate_grasp_success(",
        ):
            self.assertIn(token, inference)

    def test_historical_crog_grasp_operations_are_preserved(self):
        source = (ROOT / "utils" / "grasp_eval.py").read_text(encoding="utf-8")
        for token in (
            "shape=(480, 640)",
            "min_distance=2",
            "threshold_abs=0.4",
            "grasp_width * max_width * float(size_scale)",
            "20.0",
            "grasp_targets[:, 3] = 20",
            "grasp_targets[:, 2] = np.clip(grasp_targets[:, 2], 0, 100)",
            "if iou > iou_threshold:",
        ):
            self.assertIn(token, source)

    def test_drogoff_offset_is_trained_and_used_before_crog_scoring(self):
        dataset = (ROOT / "utils" / "dataset.py").read_text(encoding="utf-8")
        trainer = (ROOT / "train_crog.py").read_text(encoding="utf-8")
        engine = (ROOT / "engine" / "crog_engine.py").read_text(encoding="utf-8")
        model = (ROOT / "model" / "drogoff.py").read_text(encoding="utf-8")
        self.assertIn("with_grasp_offset=needs_offset", trainer)
        self.assertIn(
            'needs_offset = bool(getattr(model, "supports_offset", False))',
            trainer,
        )
        self.assertIn('data["grasp_masks"].get("off")', engine)
        self.assertIn("make_dense_offset_with_radius_np", dataset)
        self.assertIn("supports_offset = True", model)
        self.assertIn("offset_loss", model)
        self.assertIn("refine_with_offset(", engine)
        self.assertIn("resample_grasp_geometry(", engine)
        drogoff_config = (
            ROOT / "config" / "OCID-VLG" / "drogoff.yaml"
        ).read_text(encoding="utf-8")
        self.assertRegex(
            drogoff_config,
            r"(?m)^\s*offset_resample_geometry:\s*True\s*$",
        )

    def test_drogoff_inference_ablations_are_checkpoint_compatible(self):
        engine = (ROOT / "engine" / "crog_engine.py").read_text(encoding="utf-8")
        evaluator = (
            ROOT / "test_crog.py"
        ).read_text(encoding="utf-8")
        inference = engine.split("def inference_with_grasp", 1)[1]
        self.assertIn(
            'getattr(args, "use_offset_at_inference", True)', engine
        )
        self.assertIn(
            'getattr(args, "filter_grasps_by_segmentation", False)', engine
        )
        self.assertIn(
            'getattr(args, "grasp_size_activation", "sigmoid")', engine
        )
        self.assertIn(
            "return torch.clamp(prediction, 0.0, 1.0)",
            engine,
        )
        self.assertEqual(
            engine.count("_decode_grasp_size_map(grasp_wid_mask_preds, args)"), 2
        )
        self.assertIn(
            'getattr(args, "test_batch_size", 1)', evaluator
        )
        self.assertIn(
            'getattr(args, "test_workers", 1)', evaluator
        )
        self.assertIn("batch_size=test_batch_size", evaluator)
        self.assertIn("num_workers=test_workers", evaluator)
        self.assertNotIn("batch_size=1", evaluator)
        self.assertNotIn("num_workers=1", evaluator)
        self.assertLess(
            inference.index("detect_grasps("),
            inference.index("_apply_model_offset("),
        )
        self.assertLess(
            inference.index("_apply_model_offset("),
            inference.index("_filter_grasp_centres("),
        )
        self.assertLess(
            inference.index("_filter_grasp_centres("),
            inference.index("_calculate_grasp_success("),
        )

        expected = {
            "drogoff.yaml": ("True", "False", "True"),
            "drogoff_mask_filter.yaml": ("True", "True", "True"),
            "drogoff_no_offset.yaml": ("False", "False", "False"),
        }
        for name, (use_offset, mask_filter, resample) in expected.items():
            source = (ROOT / "config" / "OCID-VLG" / name).read_text(
                encoding="utf-8"
            )
            with self.subTest(config=name):
                self.assertRegex(source, r"(?m)^\s*architecture:\s*drogoff\s*$")
                self.assertRegex(
                    source,
                    rf"(?m)^\s*use_offset_at_inference:\s*{use_offset}\s*$",
                )
                self.assertRegex(
                    source,
                    rf"(?m)^\s*filter_grasps_by_segmentation:\s*{mask_filter}\s*$",
                )
                self.assertRegex(
                    source,
                    rf"(?m)^\s*offset_resample_geometry:\s*{resample}\s*$",
                )
                self.assertRegex(
                    source,
                    r"(?m)^\s*grasp_size_activation:\s*auto\s*$",
                )
                self.assertRegex(
                    source,
                    r"(?m)^\s*test_batch_size:\s*32\b",
                )
                self.assertRegex(
                    source,
                    r"(?m)^\s*test_workers:\s*2\b",
                )
                self.assertRegex(source, r"(?m)^\s*batch_size:\s*32\b")
                self.assertRegex(source, r"(?m)^\s*batch_size_val:\s*32\b")

    def test_evaluation_keeps_ground_truth_on_the_input_canvas(self):
        engine = (ROOT / "engine" / "crog_engine.py").read_text(encoding="utf-8")
        model = (ROOT / "model" / "drogoff.py").read_text(encoding="utf-8")
        self.assertLess(
            model.index("if not self.training:"),
            model.index("target_size = seg.shape[-2:]"),
        )
        self.assertEqual(engine.count("ins_mask_targets = ins_mask"), 3)
        for assignment in (
            "grasp_qua_mask_targets = grasp_qua_mask",
            "grasp_sin_mask_targets = grasp_sin_mask",
            "grasp_cos_mask_targets = grasp_cos_mask",
            "grasp_wid_mask_targets = grasp_wid_mask",
        ):
            self.assertEqual(engine.count(assignment), 2)
        self.assertNotIn("ins_mask_targets = target[0]", engine)

    def test_drog_configs_match_the_requested_global_schedule(self):
        expected_optimization = {
            "drog.yaml": ("drog", 32, 32, "0.0001", 30),
            "drogoff.yaml": ("drogoff", 32, 32, "0.0001", 30),
        }
        for name, (
            architecture,
            batch_size,
            batch_size_val,
            base_lr,
            milestone,
        ) in expected_optimization.items():
            source = (ROOT / "config" / "OCID-VLG" / name).read_text(encoding="utf-8")
            with self.subTest(config=name):
                self.assertRegex(source, rf"(?m)^\s*architecture:\s*{architecture}\s*$")
                self.assertRegex(source, r"(?m)^\s*epochs:\s*36\s*$")
                self.assertRegex(
                    source, rf"(?m)^\s*milestones:\s*\[{milestone}\]\s*$"
                )
                self.assertRegex(source, r"(?m)^\s*save_freq:\s*0\s*$")
                self.assertRegex(source, r"(?m)^\s*val_freq:\s*1\s*$")
                self.assertRegex(source, r"(?m)^\s*val_start_epoch:\s*11\s*$")
                self.assertRegex(source, r"(?m)^\s*save_epochs:\s*\[5,\s*10\]\s*$")
                self.assertRegex(source, rf"(?m)^\s*batch_size:\s*{batch_size}\b")
                self.assertRegex(
                    source,
                    rf"(?m)^\s*batch_size_val:\s*{batch_size_val}\b",
                )
                self.assertRegex(
                    source, rf"(?m)^\s*base_lr:\s*{re.escape(base_lr)}\b"
                )
                self.assertRegex(
                    source,
                    r"(?m)^\s*evaluation_protocol:\s*crog_legacy\s*$",
                )

    def test_primary_ocid_profiles_define_batchable_auto_decoding(self):
        profiles = (
            "drogoff.yaml",
            "drog.yaml",
            "crog_multiple_r50.yaml",
            "lgd.yaml",
            "ggcnnclip.yaml",
            "grconvnetclip.yaml",
            "etrg.yaml",
        )
        for name in profiles:
            source = (ROOT / "config" / "OCID-VLG" / name).read_text(
                encoding="utf-8"
            )
            with self.subTest(config=name):
                self.assertRegex(
                    source,
                    r"(?m)^\s*test_batch_size:\s*32\b",
                )
                self.assertRegex(
                    source,
                    r"(?m)^\s*test_workers:\s*2\b",
                )
                self.assertRegex(
                    source,
                    r"(?m)^\s*grasp_size_activation:\s*auto\s*$",
                )

if __name__ == "__main__":
    unittest.main()
