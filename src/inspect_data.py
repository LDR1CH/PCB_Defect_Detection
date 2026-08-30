"""Phase 1 sanity check for the DeepPCB annotations.

Two jobs:

1. Draw the ground-truth boxes of a few sample images and save the annotated
   images to ``results/data_checks/`` so the parsing can be eyeballed.
2. Scan every annotation file in both official splits and report the distinct
   `type` values found, so the integer-to-class-name mapping (and whether a +1
   shift is needed for torchvision's background-is-0 convention) is confirmed
   against the real data rather than assumed.

Run from the repo root:  python -m src.inspect_data
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless: this script only writes files, never shows a window
import matplotlib.pyplot as plt
from PIL import Image

from . import config
from .data_utils import Sample, load_split, parse_annotation_file
from .visualize import draw_boxes


def scan_annotations(samples: List[Sample], split_name: str) -> Counter:
    """Report class frequencies, box statistics and any malformed annotations.

    Returns the Counter of `type` value -> number of defect instances.
    """
    type_counts: Counter = Counter()
    defects_per_image: Counter = Counter()
    problems: List[str] = []
    widths, heights = [], []
    max_w, max_h = config.IMAGE_SIZE

    for sample in samples:
        defects = parse_annotation_file(sample.annotation_path)
        defects_per_image[len(defects)] += 1
        if not defects:
            problems.append(f"{sample.image_id}: no defects annotated")
        for x1, y1, x2, y2, defect_type in defects:
            type_counts[defect_type] += 1
            widths.append(x2 - x1)
            heights.append(y2 - y1)
            if x2 <= x1 or y2 <= y1:
                problems.append(f"{sample.image_id}: degenerate box {(x1, y1, x2, y2)}")
            if x1 < 0 or y1 < 0 or x2 > max_w or y2 > max_h:
                problems.append(f"{sample.image_id}: box out of bounds {(x1, y1, x2, y2)}")

    total_defects = sum(type_counts.values())
    print(f"\n=== {split_name}: {len(samples)} images, {total_defects} defect instances ===")
    print(f"distinct `type` values found: {sorted(type_counts)}")
    print(f"{'type':>5}  {'class name':<16} {'count':>6}  {'share':>6}")
    for defect_type in sorted(type_counts):
        count = type_counts[defect_type]
        name = config.CLASS_NAMES.get(defect_type, "!! UNKNOWN !!")
        print(f"{defect_type:>5}  {name:<16} {count:>6}  {count / total_defects:>5.1%}")

    counts_seen = sorted(defects_per_image)
    print(
        f"defects per image: min {counts_seen[0]}, max {counts_seen[-1]}, "
        f"mean {total_defects / len(samples):.1f}"
    )
    print(
        f"box size (px): width {min(widths)}-{max(widths)} (mean {sum(widths)/len(widths):.1f}), "
        f"height {min(heights)}-{max(heights)} (mean {sum(heights)/len(heights):.1f})"
    )
    if problems:
        print(f"PROBLEMS ({len(problems)}), first 10:")
        for problem in problems[:10]:
            print(f"  - {problem}")
    else:
        print("no degenerate or out-of-bounds boxes found")
    return type_counts


def check_image_sizes(samples: List[Sample], limit: int = 50) -> None:
    """Confirm the first `limit` images all have the expected resolution."""
    sizes = Counter(Image.open(s.image_path).size for s in samples[:limit])
    print(f"\nimage sizes over first {min(limit, len(samples))} images: {dict(sizes)}")


def draw_sample(sample: Sample, output_path: Path) -> None:
    """Save `sample`'s image with its ground-truth boxes and labels drawn on it."""
    image = Image.open(sample.image_path).convert("RGB")
    defects = parse_annotation_file(sample.annotation_path)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_boxes(
        ax,
        image,
        boxes=[d[:4] for d in defects],
        labels=[d[4] for d in defects],
        title=f"{sample.image_id} - {len(defects)} defects",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path.relative_to(config.PROJECT_ROOT)} ({len(defects)} boxes)")


def main() -> None:
    """Scan both splits for class IDs, then draw a few annotated samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-samples", type=int, default=4, help="how many images to visualize"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=137,
        help="spacing between visualized samples, to pull them from different groups",
    )
    args = parser.parse_args()

    trainval = load_split(config.TRAINVAL_LIST)
    test = load_split(config.TEST_LIST)

    trainval_counts = scan_annotations(trainval, "trainval")
    test_counts = scan_annotations(test, "test")

    combined = sorted(set(trainval_counts) | set(test_counts))
    print(f"\n=== combined: distinct `type` values across the whole dataset: {combined} ===")
    unknown = [t for t in combined if t not in config.CLASS_NAMES]
    print(f"minimum type value: {min(combined)} (0 is reserved for background)")
    print(
        "label shift needed for torchvision: "
        + ("NO - IDs already start at 1" if min(combined) >= 1 else "YES - IDs start at 0")
    )
    print(f"IDs not present in config.CLASS_NAMES: {unknown or 'none'}")

    check_image_sizes(trainval)

    output_dir = config.RESULTS_DIR / "data_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== drawing {args.num_samples} annotated samples ===")
    for i in range(args.num_samples):
        sample = trainval[(i * args.stride) % len(trainval)]
        draw_sample(sample, output_dir / f"gt_{sample.image_id}.png")


if __name__ == "__main__":
    main()
