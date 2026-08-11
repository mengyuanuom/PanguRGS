import hashlib
import io
import ssl
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from utils import pretrained


ROOT = Path(__file__).resolve().parents[1]


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class PretrainedWeightsTest(unittest.TestCase):
    def test_manifest_matches_toolrgsnpu(self):
        self.assertEqual(
            {item.filename for item in pretrained.ARTIFACTS.values()},
            {
                "RN50.pt",
                "RN101.pt",
                "ViT-B-16.pt",
                "dinov2_vitb14_reg4_pretrain.pth",
                "mambavision_tiny_1k.pth.tar",
                "resnet18-f37072fd.pth",
            },
        )
        for key, artifact in pretrained.ARTIFACTS.items():
            with self.subTest(key=key):
                self.assertTrue(artifact.url.startswith("https://"))
                self.assertEqual(Path(artifact.filename).name, artifact.filename)

    def test_official_clip_urls_embed_the_full_checksum(self):
        for key in ("clip-rn50", "clip-rn101", "clip-vit-b16"):
            artifact = pretrained.ARTIFACTS[key]
            with self.subTest(key=key):
                self.assertEqual(len(artifact.sha256), 64)
                self.assertIn(artifact.sha256, artifact.url)

    def test_missing_file_downloads_once_and_is_reused(self):
        payload = b"official-test-weight"
        artifact = pretrained.PretrainedArtifact(
            "test-weight.bin",
            "test weight",
            "https://example.invalid/test-weight.bin",
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / artifact.filename
            with mock.patch.dict(pretrained.ARTIFACTS, {"test": artifact}):
                with mock.patch.object(
                    pretrained.urllib.request,
                    "urlopen",
                    return_value=_Response(payload),
                ) as urlopen:
                    self.assertEqual(
                        pretrained.ensure_pretrained(target, "test"), target
                    )
                    self.assertEqual(target.read_bytes(), payload)
                    self.assertEqual(
                        pretrained.ensure_pretrained(target, "test"), target
                    )
                    urlopen.assert_called_once()
            self.assertFalse((target.with_name(target.name + ".lock")).exists())
            self.assertFalse(list(target.parent.glob("*.part.*")))

    def test_manifest_command_lists_links_without_downloading(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "download_pretrained.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        for artifact in pretrained.ARTIFACTS.values():
            self.assertIn(artifact.filename, result.stdout)
            self.assertIn(artifact.url, result.stdout)


    def test_insecure_mode_is_explicit_and_disables_verification(self):
        with mock.patch.dict(pretrained.os.environ, {}, clear=True):
            with self.assertWarns(RuntimeWarning):
                context = pretrained._download_ssl_context(insecure=True)
        self.assertFalse(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_NONE)

    def test_existing_weight_does_not_require_an_ssl_context(self):
        artifact = pretrained.ARTIFACTS["dinov2-vitb14-reg4"]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / artifact.filename
            target.write_bytes(b"existing-weight")
            with mock.patch.object(
                pretrained,
                "_download_ssl_context",
                side_effect=AssertionError("SSL context should not be created"),
            ):
                self.assertEqual(
                    pretrained.ensure_pretrained(target, "dinov2-vitb14-reg4"),
                    target,
                )


    def test_models_and_launchers_use_automatic_downloads(self):
        crog = (ROOT / "model" / "crog.py").read_text(encoding="utf-8")
        drog = (ROOT / "model" / "drog.py").read_text(encoding="utf-8")
        launcher = (ROOT / "tools" / "train_8npu.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('ensure_pretrained(cfg.clip_pretrain, "clip-rn50")', crog)
        self.assertIn('ensure_pretrained(cfg.clip_pretrain, "clip-vit-b16")', drog)
        self.assertIn('cfg.dino_pretrain, "dinov2-vitb14-reg4"', drog)
        self.assertIn("tools/download_pretrained.py clip-vit-b16", launcher)
        self.assertIn("tools/download_pretrained.py dinov2-vitb14-reg4", launcher)
        self.assertIn("tools/download_pretrained.py resnet18", launcher)


if __name__ == "__main__":
    unittest.main()
