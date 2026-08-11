from collections import Counter
from pathlib import Path
import random
import sys
from types import SimpleNamespace

from tools.dataset_converters.grasp_tools.augment import (
    LanguageScheduler,
    SourceObject,
    balanced_quotas,
    balanced_category_candidate,
    balanced_scene_sizes,
    build_config,
    parse_args,
    placement_scale_backoff,
    plan_query_targets,
    plan_split_scenes,
)
from utils.grasp_tool_language import CANONICAL_CATEGORY_NAMES


def fake_sources():
    result = {}
    for category in CANONICAL_CATEGORY_NAMES:
        result[category] = [
            SourceObject(
                source_id=f"{category}:{index}",
                image_path=Path(f"{category}_{index}.jpg"),
                object_index=index,
                category_key=category,
                category_name=CANONICAL_CATEGORY_NAMES[category],
                mask=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
                grasps=(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),),
            )
            for index in range(3)
        ]
    return result


def delta(values):
    values = list(values)
    return max(values) - min(values)


def test_default_cli_is_balanced_difficulty_one(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["augment.py"])
    config = build_config(parse_args())
    assert config.src_dir == "assets/grasp_tools/graspall"
    assert config.background_dir == "assets/grasp_tools/backgrounds"
    assert config.out_dir == "datasets/grasp-tools/aug_graspall_v2"
    assert (config.train_scenes, config.val_scenes, config.test_scenes) == (
        6000, 500, 1000
    )
    assert (config.objects_min, config.objects_max) == (2, 3)
    assert config.train_queries_per_scene == 4
    assert config.eval_queries_per_scene == 4
    assert config.max_query_difficulty == 1
    assert config.language_templates == "shared"
    assert config.same_category_probability == 0.0
    assert config.hard_negative_probability == 0.0

    monkeypatch.setattr(sys, "argv", ["augment.py", "--smoke-test"])
    smoke_config = build_config(parse_args())
    assert (
        smoke_config.train_scenes,
        smoke_config.val_scenes,
        smoke_config.test_scenes,
    ) == (22, 22, 22)


def test_category_query_marks_dynamic_prompt_cycle():
    objects = [
        {"category": "wrench", "bbox": [0, 0, 10, 10]},
        {"category": "pliers", "bbox": [20, 0, 30, 10]},
    ]
    candidate = balanced_category_candidate(objects, 0)
    query = LanguageScheduler(random.Random(7)).render(
        candidate,
        objects,
        split="train",
        scene_id="scene_000001",
        query_index=0,
        language_templates="shared",
        category_vocabulary="expanded",
    )
    assert query["type"] == "category"
    assert query["prompt_cycle"] == "category_v1"


def test_placement_scale_backoff_is_bounded_and_monotonic():
    values = [placement_scale_backoff(attempt) for attempt in range(30)]
    assert values[0] == 1.0
    assert all(current >= following for current, following in zip(values, values[1:]))
    assert min(values) == 0.55
    assert placement_scale_backoff(100) == 0.55


def test_balanced_integer_quotas_and_scene_sizes():
    rng = random.Random(7)
    quotas = balanced_quotas(101, list("abcdef"), rng)
    assert sum(quotas.values()) == 101
    assert delta(quotas.values()) <= 1

    sizes = balanced_scene_sizes(800, 3, 5, rng)
    assert len(sizes) == 800
    assert min(sizes) == 3
    assert max(sizes) == 5
    assert sum(sizes) == 3200
    counts = Counter(sizes)
    assert delta(counts.values()) <= 1


def test_split_planner_balances_categories_sources_and_queries():
    sources = fake_sources()
    config = SimpleNamespace(
        objects_min=3,
        objects_max=5,
        same_category_probability=0.35,
        hard_negative_probability=0.30,
    )
    rng = random.Random(2025)
    scenes, source_usage = plan_split_scenes(66, sources, config, rng)

    placements = Counter(
        source.category_key for scene in scenes for source in scene
    )
    assert len(scenes) == 66
    assert sum(placements.values()) == 264
    assert delta(placements[category] for category in sources) <= 1

    for category, category_sources in sources.items():
        assert delta(
            source_usage.get(source.source_id, 0)
            for source in category_sources
        ) <= 1

    targets, target_quota = plan_query_targets(scenes, 6, rng)
    assert all(len(scene_targets) == 6 for scene_targets in targets)
    assert sum(target_quota.values()) == 396
    assert delta(target_quota.values()) <= 1
    for scene, scene_targets in zip(scenes, targets):
        assert all(0 <= target < len(scene) for target in scene_targets)
