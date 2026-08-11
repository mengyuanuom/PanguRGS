"""Dataset adapter for the generated Grasp-Tools schema-v2.1 data."""

import functools
import json
import os

import cv2
import numpy as np
import torch
from skimage.draw import polygon as rasterize_polygon
from skimage.filters import gaussian
from torch.utils.data import Dataset

from .dataset import make_dense_offset_with_radius_np, tokenize
from .grasp_tool_language import category_prompt_for_epoch


class GraspToolTransforms:
    """Convert four-point grasps and rasterize CROG-compatible targets."""

    def __init__(self, width_factor=100.0, width=416, height=416):
        self.width_factor = float(width_factor)
        self.width = int(width)
        self.height = int(height)

    def __call__(self, grasp_rectangles, target):
        result = []
        for rect in grasp_rectangles:
            (cx, cy), (width, height), angle = cv2.minAreaRect(
                np.asarray(rect, dtype=np.float32)
            )
            if width < height:
                width, height = height, width
                angle += 90.0
            while angle >= 90.0:
                angle -= 180.0
            while angle < -90.0:
                angle += 180.0
            result.append([cx, cy, width, height, angle, target])
        return np.asarray(result, dtype=np.float32).reshape(-1, 6)

    def generate_masks(self, grasp_rectangles, include_short=False):
        quality = np.zeros((self.height, self.width), dtype=np.float32)
        angle_map = np.zeros_like(quality)
        width_map = np.zeros_like(quality)
        short_map = np.zeros_like(quality) if include_short else None
        for center_x, center_y, width, height, angle, _ in grasp_rectangles:
            rect = (
                (float(center_x), float(center_y)),
                (float(width) / 2.0, float(height)),
                -(float(angle) + 180.0),
            )
            box = cv2.boxPoints(rect)
            rr, cc = rasterize_polygon(
                box[:, 1], box[:, 0], shape=(self.height, self.width)
            )
            quality[rr, cc] = 1.0
            angle_map[rr, cc] = angle + 180.0 if angle < 0 else angle
            width_map[rr, cc] = np.clip(
                width, 0.0, self.width_factor
            ) / self.width_factor
            if short_map is not None:
                short_map[rr, cc] = np.clip(
                    height, 0.0, self.width_factor
                ) / self.width_factor
        masks = {
            "qua": gaussian(quality, 3, preserve_range=True).astype(np.float32),
            "ang": np.deg2rad(angle_map).astype(np.float32),
            "wid": gaussian(width_map, 3, preserve_range=True).astype(np.float32),
        }
        if short_map is not None:
            masks["short"] = gaussian(
                short_map, 3, preserve_range=True
            ).astype(np.float32)
        return masks


class GraspToolDataset(Dataset):
    """Expose every language query as one referring-grasp training sample."""

    def __init__(
        self,
        root_dir,
        input_size=416,
        split="train",
        word_length=32,
        with_offset=False,
        with_short_side=False,
        offset_radius=20.0,
        offset_sigma=None,
        dynamic_train_prompts=False,
        dynamic_prompt_seed=2025,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.split = str(split)
        self.epoch = 1
        self.input_size = (int(input_size), int(input_size))
        self.word_length = int(word_length)
        self.with_offset = bool(with_offset)
        self.with_short_side = bool(with_short_side)
        self.offset_radius = float(offset_radius)
        self.offset_sigma = offset_sigma
        self.dynamic_train_prompts = bool(dynamic_train_prompts)
        self.dynamic_prompt_seed = int(dynamic_prompt_seed)
        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073]
        ).reshape(3, 1, 1)
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711]
        ).reshape(3, 1, 1)
        self.grasp_transform = GraspToolTransforms(
            width_factor=100.0,
            width=self.input_size[1],
            height=self.input_size[0],
        )

        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Grasp-Tools split directory not found: {split_dir}"
            )
        self.samples = self._build_index(split_dir)
        if not self.samples:
            raise RuntimeError(f"No Grasp-Tools samples found in: {split_dir}")

    def _build_index(self, split_dir):
        samples = []
        index_path = os.path.join(split_dir, "index.jsonl")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        samples.append((
                            os.path.join(split_dir, row["image"]),
                            os.path.join(split_dir, row["annotation"]),
                            int(row["query_index"]),
                        ))
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"Invalid {index_path}:{line_number}: {error}"
                        ) from error
            return samples

        for filename in sorted(os.listdir(split_dir)):
            if not filename.endswith(".json"):
                continue
            json_path = os.path.join(split_dir, filename)
            annotation = self._read_annotation(json_path)
            image_name = annotation.get("image_filename")
            if not image_name:
                stem = os.path.splitext(filename)[0]
                image_name = next(
                    (
                        stem + suffix
                        for suffix in (".jpg", ".png", ".jpeg")
                        if os.path.isfile(os.path.join(split_dir, stem + suffix))
                    ),
                    stem + ".png",
                )
            image_path = os.path.join(split_dir, image_name)
            queries = annotation.get("queries") or []
            if queries:
                samples.extend(
                    (image_path, json_path, query_index)
                    for query_index in range(len(queries))
                )
            else:
                samples.append((image_path, json_path, None))
        return samples

    @functools.lru_cache(maxsize=512)
    def _read_annotation(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = max(1, int(epoch))

    @staticmethod
    def _transform_matrix(image_size, input_size):
        ori_h, ori_w = image_size
        inp_h, inp_w = input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x = (inp_w - new_w) / 2.0
        bias_y = (inp_h - new_h) / 2.0
        src = np.asarray([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.asarray(
            [[bias_x, bias_y], [new_w + bias_x, bias_y],
             [bias_x, new_h + bias_y]],
            np.float32,
        )
        return cv2.getAffineTransform(src, dst), cv2.getAffineTransform(dst, src)

    def __getitem__(self, index):
        image_path, json_path, query_index = self.samples[index]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read Grasp-Tools image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = image.shape[:2]
        data = self._read_annotation(json_path)
        objects = data.get("objects") or []

        if query_index is None:
            target_idx = 0
            obj = objects[target_idx]
            sentence = str(obj.get("language", obj.get("category", "")))
            sent_id = os.path.basename(json_path)
            query_type, difficulty, program = "legacy", 1, []
        else:
            queries = data.get("queries") or []
            if not 0 <= query_index < len(queries):
                raise IndexError(f"Invalid query {query_index} in {json_path}")
            query = queries[query_index]
            target_idx = int(query["target_idx"])
            if not 0 <= target_idx < len(objects):
                raise IndexError(f"Invalid target {target_idx} in {json_path}")
            obj = objects[target_idx]
            sentence = str(query["text"])
            sent_id = query.get(
                "query_id",
                f"{os.path.splitext(os.path.basename(json_path))[0]}_q{query_index:02d}",
            )
            query_type = query.get("type", "unknown")
            difficulty = int(query.get("difficulty", 0))
            program = query.get("program", [])
            if (
                self.dynamic_train_prompts
                and self.split == "train"
                and query_type == "category"
                # Schema-v2.1 files generated before prompt_cycle was added
                # are safe to rotate too: type=category is emitted only for a
                # unique-category target. Keep rejecting unknown future cycles.
                and query.get("prompt_cycle", "category_v1") == "category_v1"
            ):
                sample_key = (
                    f"{data.get('scene_id', os.path.basename(json_path))}:"
                    f"{target_idx}:{sent_id}"
                )
                sentence = category_prompt_for_epoch(
                    obj["category"], sample_key, self.epoch,
                    seed=self.dynamic_prompt_seed,
                )

        mask_polygon = np.asarray(obj["mask"], dtype=np.int32)
        mask = np.zeros((ori_h, ori_w), dtype=np.uint8)
        cv2.drawContours(mask, [mask_polygon], -1, 1, -1)
        grasps = np.asarray(obj.get("grasps", []), dtype=np.float32)
        if grasps.size:
            grasps = grasps.reshape(-1, 4, 2)
        else:
            grasps = np.zeros((0, 4, 2), dtype=np.float32)

        matrix, inverse = self._transform_matrix(
            (ori_h, ori_w), self.input_size
        )
        image = cv2.warpAffine(
            image, matrix, self.input_size, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image.div_(255.0).sub_(self.mean).div_(self.std)
        mask = cv2.warpAffine(
            mask, matrix, self.input_size, flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        if len(grasps):
            homogeneous = np.concatenate(
                [grasps, np.ones((*grasps.shape[:2], 1), dtype=np.float32)],
                axis=-1,
            )
            transformed = homogeneous @ matrix.T
        else:
            transformed = np.zeros((0, 4, 2), dtype=np.float32)
        original_grasps = self.grasp_transform(grasps, target_idx)
        transformed_grasps = self.grasp_transform(transformed, target_idx)
        raw_masks = self.grasp_transform.generate_masks(
            transformed_grasps, include_short=self.with_short_side
        )
        grasp_masks = {
            "qua": torch.from_numpy(raw_masks["qua"]).float(),
            "sin": torch.from_numpy(np.sin(2.0 * raw_masks["ang"])).float(),
            "cos": torch.from_numpy(np.cos(2.0 * raw_masks["ang"])).float(),
            "wid": torch.from_numpy(raw_masks["wid"]).float(),
        }
        if self.with_short_side:
            grasp_masks["short"] = torch.from_numpy(raw_masks["short"]).float()
        if self.with_offset:
            centers = (
                transformed_grasps[:, :2]
                if len(transformed_grasps)
                else np.zeros((0, 2), dtype=np.float32)
            )
            offset, offset_weight = make_dense_offset_with_radius_np(
                centers, self.input_size, self.offset_radius,
                use_gaussian=True, sigma=self.offset_sigma,
            )
            grasp_masks["off"] = torch.from_numpy(offset).float()
            grasp_masks["off_w"] = torch.from_numpy(offset_weight).float()

        word_vec = tokenize(sentence, self.word_length, True).squeeze(0)
        if len(word_vec) < self.word_length:
            word_vec = torch.cat([
                word_vec,
                torch.zeros(self.word_length - len(word_vec), dtype=torch.long),
            ])
        elif len(word_vec) > self.word_length:
            word_vec = word_vec[:self.word_length]
        scene_id = data.get(
            "scene_id", os.path.splitext(os.path.basename(json_path))[0]
        )
        return {
            "img": image,
            "depth": torch.zeros(1, *self.input_size),
            "mask": torch.from_numpy(mask).float(),
            "grasp_masks": grasp_masks,
            "word_vec": word_vec.long(),
            "grasps": original_grasps,
            "target": obj["category"],
            "sentence": sentence,
            "bbox": obj.get("bbox"),
            "target_idx": target_idx,
            "sent_id": sent_id,
            "scene_id": scene_id,
            "query_type": query_type,
            "difficulty": difficulty,
            "program": program,
            "inverse": inverse,
            "ori_size": np.asarray([ori_h, ori_w]),
            "img_path": image_path,
        }

    @staticmethod
    def collate_fn(batch):
        grasp_masks = {
            key: torch.stack([sample["grasp_masks"][key] for sample in batch])
            for key in ("qua", "sin", "cos", "wid")
        }
        for key in ("off", "off_w"):
            if all(key in sample["grasp_masks"] for sample in batch):
                grasp_masks[key] = torch.stack(
                    [sample["grasp_masks"][key] for sample in batch]
                )
        if all("short" in sample["grasp_masks"] for sample in batch):
            grasp_masks["short"] = torch.stack(
                [sample["grasp_masks"]["short"] for sample in batch]
            )
        return {
            "img": torch.stack([sample["img"] for sample in batch]),
            "depth": torch.stack([sample["depth"] for sample in batch]),
            "mask": torch.stack([sample["mask"] for sample in batch]),
            "grasp_masks": grasp_masks,
            "word_vec": torch.stack([sample["word_vec"] for sample in batch]),
            "grasps": [sample["grasps"] for sample in batch],
            "target": [sample["target"] for sample in batch],
            "sentence": [sample["sentence"] for sample in batch],
            "bbox": [sample["bbox"] for sample in batch],
            "target_idx": [sample["target_idx"] for sample in batch],
            "sent_id": [sample["sent_id"] for sample in batch],
            "scene_id": [sample["scene_id"] for sample in batch],
            "query_type": [sample["query_type"] for sample in batch],
            "difficulty": [sample["difficulty"] for sample in batch],
            "program": [sample["program"] for sample in batch],
            "inverse": [sample["inverse"] for sample in batch],
            "ori_size": [sample["ori_size"] for sample in batch],
            "img_path": [sample["img_path"] for sample in batch],
        }
