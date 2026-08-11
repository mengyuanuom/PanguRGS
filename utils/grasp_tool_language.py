"""Deterministic language curriculum for Grasp-Tools category queries."""

from hashlib import sha256
import random


CANONICAL_CATEGORY_NAMES = {
    "tape measure": "tape measure",
    "t-hex key": "T-hex key",
    "l-hex key": "L-hex key",
    "marker": "marker",
    "wrench": "wrench",
    "pliers": "pliers",
    "mallet": "mallet",
    "screwdriver": "screwdriver",
    "clamps": "clamps",
    "spool": "spool",
    "sponge": "sponge",
    "clip": "clip",
    "crimp tool": "crimp tool",
    "screw": "screw",
    "tape": "tape",
    "box": "box",
    "nut": "nut",
    "ruler": "ruler",
    "file": "file",
    "stapler": "stapler",
    "scissors": "scissors",
    "cable": "cable",
}

# These surface forms never change the annotation's canonical class identity.
CATEGORY_DESCRIPTION_VARIANTS = {
    "tape measure": (
        "tape measure",
        "measuring tape",
        "retractable tape measure",
        "measuring tape tool",
    ),
    "t-hex key": (
        "T-hex key",
        "T-handle hex key",
        "T-shaped Allen key",
        "T-handle Allen wrench",
    ),
    "l-hex key": (
        "L-hex key",
        "L-shaped hex key",
        "L-shaped Allen key",
        "L-shaped Allen wrench",
    ),
    "marker": ("marker", "marker pen", "felt-tip marker", "felt pen"),
    "wrench": ("wrench", "spanner", "open-end wrench", "hand wrench"),
    "pliers": ("pliers", "pair of pliers", "gripping pliers", "hand pliers"),
    "mallet": ("mallet", "hand mallet", "striking mallet", "manual mallet"),
    "screwdriver": (
        "screwdriver",
        "screw driver",
        "hand screwdriver",
        "manual screwdriver",
    ),
    "clamps": ("clamps", "clamp", "clamping tool", "workpiece clamp"),
    "spool": ("spool", "reel", "thread spool", "winding spool"),
    "sponge": ("sponge", "cleaning sponge", "scrubbing sponge", "foam sponge"),
    "clip": ("clip", "fastening clip", "small clip", "holding clip"),
    "crimp tool": ("crimp tool", "crimper", "crimping tool", "wire crimper"),
    "screw": ("screw", "metal screw", "threaded screw", "fastening screw"),
    "tape": ("tape", "adhesive tape", "tape roll", "roll of tape"),
    "box": ("box", "container box", "small box", "storage box"),
    "nut": ("nut", "hex nut", "threaded nut", "metal nut"),
    "ruler": ("ruler", "measuring ruler", "straight ruler", "measurement ruler"),
    "file": ("file", "hand file", "metal file", "filing tool"),
    "stapler": ("stapler", "office stapler", "stapling tool", "desktop stapler"),
    "scissors": ("scissors", "pair of scissors", "cutting scissors", "shears"),
    "cable": ("cable", "wire", "electrical cable", "connecting cable"),
}

COMMAND_TEMPLATES = {
    "train": (
        "Pick up {description}.",
        "Grasp {description}.",
        "Please pick up {description}.",
        "Grab {description}, please.",
        "I would like you to grasp {description}.",
        "Please select {description}.",
        "Pick {description} up.",
        "Please grasp {description}.",
        "Please grab {description}.",
        "Select {description}.",
        "Choose {description}.",
        "Lift {description}.",
        "Take hold of {description}.",
        "Reach for and grasp {description}.",
        "Find and grasp {description}.",
        "Find and pick up {description}.",
        "Locate and grasp {description}.",
        "Retrieve {description}.",
        "Take {description}.",
        "Get {description}.",
        "Could you pick up {description}?",
        "Can you grasp {description}?",
    ),
    "eval": (
        "Identify and lift {description}.",
        "Move your gripper to {description}.",
        "Secure {description} with the gripper.",
        "Target {description} and lift it.",
    ),
}

if set(COMMAND_TEMPLATES["train"]) & set(COMMAND_TEMPLATES["eval"]):
    raise RuntimeError("Training and evaluation command templates must be disjoint")

if set(CATEGORY_DESCRIPTION_VARIANTS) != set(CANONICAL_CATEGORY_NAMES):
    raise RuntimeError("Every canonical Grasp-Tools category needs prompt variants")
if any(len(values) != 4 for values in CATEGORY_DESCRIPTION_VARIANTS.values()):
    raise RuntimeError("Each Grasp-Tools category must expose four prompt variants")


def canonical_category_key(value):
    """Resolve a generated canonical display name to its stable category key."""
    normalized = str(value).strip().lower().replace("_", " ")
    if normalized in CANONICAL_CATEGORY_NAMES:
        return normalized
    for key, display_name in CANONICAL_CATEGORY_NAMES.items():
        if normalized == display_name.lower():
            return key
    raise KeyError(f"Unknown Grasp-Tools category: {value!r}")


def category_prompt_pool(category):
    """Return all 88 command/term combinations for one canonical category."""
    key = canonical_category_key(category)
    return tuple(
        template.format(description=f"the {term}")
        for template in COMMAND_TEMPLATES["train"]
        for term in CATEGORY_DESCRIPTION_VARIANTS[key]
    )


def _prompt_permutation(sample_key, cycle, pool_size, seed):
    material = f"{int(seed)}:{sample_key}:{int(cycle)}".encode("utf-8")
    stable_seed = int.from_bytes(sha256(material).digest()[:8], "big", signed=False)
    order = list(range(pool_size))
    random.Random(stable_seed).shuffle(order)
    return order


def category_prompt_for_epoch(category, sample_key, epoch, seed=2025):
    """Pick from a reproducibly shuffled, non-repeating 88-prompt cycle."""
    pool = category_prompt_pool(category)
    epoch_index = max(0, int(epoch) - 1)
    cycle, position = divmod(epoch_index, len(pool))
    order = _prompt_permutation(sample_key, cycle, len(pool), seed)
    if cycle > 0:
        previous = _prompt_permutation(sample_key, cycle - 1, len(pool), seed)
        if order[0] == previous[-1]:
            order[0], order[1] = order[1], order[0]
    return pool[order[position]]
