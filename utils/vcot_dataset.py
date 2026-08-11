"""VCoT/Grasp-Anything dataset adapter for CROG-NPU.

The split CSV is lightweight metadata. Images, grasp tensors, and masks are
loaded lazily, one sample at a time, from the Grasp-Anything dataset root.
"""

import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import GraspTransforms, make_dense_offset_with_radius_np, tokenize


from .vcot_geometry import (
    grasp_anything_to_quads,
    resolve_vcot_grasp_root,
    resolve_vcot_split,
)

def load_vcot_grasps(path):
    """Load one VCoT ``.pt`` annotation on CPU and return quads and scores."""
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0/2.1 do not expose weights_only.
        tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return grasp_anything_to_quads(tensor)


class VCoTDataset(Dataset):
    """Adapt VCoT split metadata and Grasp-Anything files to CROG-NPU."""

    def __init__(
        self,
        root_dir,
        input_size=416,
        split="train",
        word_length=17,
        split_root=None,
        prompt_template="Grasp the {object_name}",
        with_offset=False,
        offset_radius=20.0,
        offset_sigma=None,
        grasp_size_factor=100.0,
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.input_size = (int(input_size), int(input_size))
        self.word_length = int(word_length)
        self.prompt_template = str(prompt_template)
        self.with_offset = bool(with_offset)
        self.offset_radius = float(offset_radius)
        self.offset_sigma = offset_sigma
        self.grasp_size_factor = float(grasp_size_factor)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
        self.grasp_transform = GraspTransforms(
            width_factor=self.grasp_size_factor, width=self.input_size[1], height=self.input_size[0]
        )

        if split_root is None:
            split_root = Path(__file__).resolve().parents[1] / "split" / "vcot"
        self.split_root = Path(split_root).expanduser()
        self.csv_path = self.split_root / resolve_vcot_split(split)
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"VCoT split CSV not found: {self.csv_path}. "
                "Set DATA.split_root or keep split/vcot inside the VCoT dataset root."
            )
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Grasp-Anything root not found: {self.root_dir}")
        self.grasp_root = resolve_vcot_grasp_root(self.root_dir)
        if not self.grasp_root.is_dir():
            raise FileNotFoundError(
                "VCoT positive grasp directory not found; expected "
                f"{self.root_dir / 'grasp_label_positive'} (official) or "
                f"{self.root_dir / 'positive_grasp'} (legacy)."
            )

        self.samples = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as stream:
            for row_number, row in enumerate(csv.reader(stream), start=1):
                if not row:
                    continue
                if len(row) < 3:
                    raise ValueError(
                        f"Malformed VCoT CSV row {row_number} in {self.csv_path}: {row!r}"
                    )
                grasp_id = row[0].strip()
                object_name = row[1].strip()
                description = ",".join(row[2:]).strip()
                self.samples.append((grasp_id, object_name, description, row_number))
        if not self.samples:
            raise ValueError(f"VCoT split contains no samples: {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _require(path, kind, grasp_id):
        if not path.is_file():
            raise FileNotFoundError(f"Missing VCoT {kind} for {grasp_id}: {path}")
        return path

    @staticmethod
    def _transform_matrix(image_size, input_size):
        ori_h, ori_w = image_size
        inp_h, inp_w = input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2.0, (inp_h - new_h) / 2.0
        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], dtype=np.float32)
        dst = np.array(
            [[bias_x, bias_y], [new_w + bias_x, bias_y], [bias_x, new_h + bias_y]],
            dtype=np.float32,
        )
        return cv2.getAffineTransform(src, dst), cv2.getAffineTransform(dst, src)

    @staticmethod
    def _apply_affine(quads, matrix):
        if not len(quads):
            return np.zeros((0, 4, 2), dtype=np.float32)
        ones = np.ones((*quads.shape[:2], 1), dtype=np.float32)
        homogeneous = np.concatenate([quads, ones], axis=-1)
        return np.einsum("ij,nkj->nki", matrix, homogeneous).astype(np.float32)

    @staticmethod
    def _bbox_from_mask(mask):
        ys, xs = np.nonzero(mask > 0)
        if not len(xs):
            return None
        return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

    def __getitem__(self, index):
        grasp_id, object_name, description, row_number = self.samples[index]
        scene_id = grasp_id.rsplit("_", 1)[0]
        image_path = self._require(
            self.root_dir / "image" / f"{scene_id}.jpg", "image", grasp_id
        )
        grasp_path = self._require(
            self.grasp_root / f"{grasp_id}.pt", "grasp", grasp_id
        )
        mask_path = self._require(
            self.root_dir / "mask" / f"{grasp_id}.npy", "mask", grasp_id
        )

        image = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode VCoT image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = image.shape[:2]

        mask = np.asarray(np.load(mask_path, allow_pickle=False)).squeeze()
        if mask.ndim != 2:
            raise ValueError(f"VCoT mask must be 2-D, got {mask.shape}: {mask_path}")
        if tuple(mask.shape) != (ori_h, ori_w):
            raise ValueError(
                f"VCoT mask/image size mismatch for {grasp_id}: "
                f"mask={mask.shape}, image={(ori_h, ori_w)}"
            )
        mask = (mask > 0).astype(np.uint8)

        quads, grasp_scores = load_vcot_grasps(grasp_path)
        if not len(quads):
            raise ValueError(f"VCoT sample has no valid positive grasps: {grasp_path}")

        matrix, inverse = self._transform_matrix((ori_h, ori_w), self.input_size)
        image = cv2.warpAffine(
            image,
            matrix,
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image.div_(255.0).sub_(self.mean).div_(self.std)
        resized_mask = cv2.warpAffine(
            mask,
            matrix,
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        transformed_quads = self._apply_affine(quads, matrix)
        original_targets = self.grasp_transform(quads, target=0)
        input_targets = self.grasp_transform(transformed_quads, target=0)
        raw_masks = self.grasp_transform.generate_masks(input_targets)
        angle = raw_masks["ang"].astype(np.float32) * np.pi / 180.0
        grasp_masks = {
            "qua": torch.from_numpy(raw_masks["qua"].astype(np.float32) / 255.0),
            "sin": torch.from_numpy(np.sin(2.0 * angle)).float(),
            "cos": torch.from_numpy(np.cos(2.0 * angle)).float(),
            "wid": torch.from_numpy(raw_masks["wid"].astype(np.float32) / 255.0),
            "short": torch.from_numpy(
                raw_masks["short"].astype(np.float32) / 255.0
            ),
        }
        if self.with_offset:
            offsets, offset_weights = make_dense_offset_with_radius_np(
                centers_xy=input_targets[:, :2],
                img_size_hw=self.input_size,
                r_pix=self.offset_radius,
                use_gaussian=True,
                sigma=self.offset_sigma,
            )
            grasp_masks["off"] = torch.from_numpy(offsets).float()
            grasp_masks["off_w"] = torch.from_numpy(offset_weights).float()

        sentence = self.prompt_template.format(
            object_name=object_name, object=object_name, description=description
        )
        word_vec = tokenize(sentence, self.word_length, True).squeeze(0).long()
        if word_vec.numel() < self.word_length:
            word_vec = torch.cat(
                [word_vec, torch.zeros(self.word_length - word_vec.numel(), dtype=torch.long)]
            )
        else:
            word_vec = word_vec[: self.word_length]

        return {
            "img": image,
            "mask": torch.from_numpy(resized_mask).float(),
            "grasp_masks": grasp_masks,
            "word_vec": word_vec,
            # Evaluation predictions are inverse-warped, so targets stay in original pixels.
            "grasps": original_targets,
            "grasp_scores": grasp_scores,
            "target": object_name,
            "sentence": sentence,
            "description": description,
            "object_name": object_name,
            "bbox": self._bbox_from_mask(resized_mask),
            "target_idx": 0,
            "sent_id": grasp_id,
            "scene_id": scene_id,
            "grasp_id": grasp_id,
            "csv_row": row_number,
            "csv_path": os.fspath(self.csv_path),
            "inverse": inverse,
            "ori_size": np.array([ori_h, ori_w]),
            "img_path": os.fspath(image_path),
        }

    @staticmethod
    def collate_fn(batch):
        grasp_masks = {
            key: torch.stack([sample["grasp_masks"][key] for sample in batch])
            for key in ("qua", "sin", "cos", "wid", "short")
        }
        for key in ("off", "off_w"):
            if all(key in sample["grasp_masks"] for sample in batch):
                grasp_masks[key] = torch.stack(
                    [sample["grasp_masks"][key] for sample in batch]
                )

        stacked = {
            "img": torch.stack([sample["img"] for sample in batch]),
            "mask": torch.stack([sample["mask"] for sample in batch]),
            "grasp_masks": grasp_masks,
            "word_vec": torch.stack([sample["word_vec"] for sample in batch]),
        }
        list_keys = (
            "grasps", "grasp_scores", "target", "sentence", "description",
            "object_name", "bbox", "target_idx", "sent_id", "scene_id",
            "grasp_id", "csv_row", "csv_path", "inverse", "ori_size", "img_path",
        )
        stacked.update({key: [sample[key] for sample in batch] for key in list_keys})
        return stacked
