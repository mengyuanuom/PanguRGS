from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


PANGU_MODELS = {
    "crog": "PanguCROG",
    "drog": "PanguDROG",
    "drogoff": "PanguDROGOFF",
    "ssg": "PanguSSG",
}

PANGU_VARIANTS = {
    "crogoff": "PanguCROGOFF",
    "etrg": "PanguETRG",
    "ggcnnclip": "PanguGGCNNCLIP",
    "grconvnetclip": "PanguGRConvNetCLIP",
    "graspmamba": "PanguGraspMamba",
    "lgd": "PanguLGD",
    "maplegrasp": "PanguMapleGrasp",
}


class PanguNamespaceTest(unittest.TestCase):
    def test_pangu_public_modules_keep_original_model_names(self):
        for model_name, class_name in PANGU_MODELS.items():
            with self.subTest(model=model_name):
                source = (ROOT / "model" / f"pangu_{model_name}.py").read_text(
                    encoding="utf-8"
                )
                self.assertIn(f"class {class_name}", source)
                self.assertIn(f"from .{model_name} import", source)

        for model_name, class_name in PANGU_VARIANTS.items():
            with self.subTest(model=model_name):
                source = (
                    ROOT / "model" / "toolrgs" / f"pangu_{model_name}.py"
                ).read_text(encoding="utf-8")
                self.assertIn(f"class {class_name}", source)
                self.assertIn(f"from .{model_name} import", source)

    def test_model_factories_accept_new_and_legacy_names(self):
        root_registry = (ROOT / "model" / "__init__.py").read_text(
            encoding="utf-8"
        )
        variant_registry = (
            ROOT / "model" / "toolrgs" / "__init__.py"
        ).read_text(encoding="utf-8")
        for model_name in ("crog", "drog", "drogoff"):
            self.assertIn(f'"{model_name}"', root_registry)
            self.assertIn(f'"pangu_{model_name}"', root_registry)
        for model_name in PANGU_VARIANTS:
            self.assertIn(f'"{model_name}"', variant_registry)
            self.assertIn(f'"pangu_{model_name}"', variant_registry)

    def test_pangu_configs_preserve_dataset_and_training_contracts(self):
        pangu_paths = sorted((ROOT / "config").rglob("pangu_*.yaml"))
        self.assertEqual(len(pangu_paths), 32)
        for pangu_path in pangu_paths:
            with self.subTest(config=pangu_path.as_posix()):
                legacy_path = pangu_path.with_name(
                    pangu_path.name.removeprefix("pangu_")
                )
                pangu_cfg = yaml.safe_load(
                    pangu_path.read_text(encoding="utf-8")
                )
                legacy_cfg = yaml.safe_load(
                    legacy_path.read_text(encoding="utf-8")
                )
                self.assertEqual(pangu_cfg["DATA"], legacy_cfg["DATA"])

                pangu_arch = pangu_cfg.get("MODEL", {}).get("architecture")
                if pangu_arch is not None:
                    legacy_arch = legacy_cfg.get("MODEL", {}).get(
                        "architecture"
                    )
                    if legacy_arch is None:
                        legacy_arch = "ssg" if "ssg" in legacy_path.stem else "crog"
                    self.assertEqual(pangu_arch, f"pangu_{legacy_arch}")

                pangu_train = dict(pangu_cfg["TRAIN"])
                legacy_train = dict(legacy_cfg["TRAIN"])
                if "exp_name" in legacy_train:
                    self.assertEqual(
                        pangu_train.pop("exp_name"),
                        f"pangu_{legacy_train.pop('exp_name')}",
                    )
                if pangu_train.get("weight"):
                    self.assertEqual(
                        pangu_train["weight"],
                        legacy_train["weight"].replace(
                            "/maplegrasp_stage1_",
                            "/pangu_maplegrasp_stage1_",
                        ),
                    )
                    pangu_train["weight"] = legacy_train["weight"]
                self.assertEqual(pangu_train, legacy_train)
                self.assertEqual(pangu_cfg.get("TEST"), legacy_cfg.get("TEST"))


if __name__ == "__main__":
    unittest.main()
