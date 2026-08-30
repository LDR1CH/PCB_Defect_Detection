"""Evaluate the trained detector on the official DeepPCB test set.

Reports COCO-style mAP@[0.5:0.95], mAP@0.5 and per-class AP via torchmetrics,
plus AP at DeepPCB's own IoU>0.33 benchmark threshold, and saves prediction-vs-
ground-truth images so failures a single number would hide stay visible.

Run from the repo root:  python -m src.evaluate
"""

import argparse
import json
from typing import Dict

import torch
from torchmetrics.detection import MeanAveragePrecision

from . import config
from .dataset import build_dataloader, build_datasets
from .model import load_checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run inference over `loader`, returning matched prediction/target lists.

    Predictions are returned unthresholded: mAP integrates over the full
    precision-recall curve, so filtering by score first would truncate it.
    """
    model.eval()
    all_predictions, all_targets = [], []

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            all_predictions.append({k: v.detach().cpu() for k, v in output.items()})
            all_targets.append(
                {
                    "boxes": target["boxes"].detach().cpu(),
                    "labels": target["labels"].detach().cpu(),
                }
            )
        if batch_index % 10 == 0:
            print(f"  {len(all_predictions)} images done", flush=True)

    return all_predictions, all_targets


def compute_metrics(predictions, targets) -> Dict:
    """Compute COCO mAP (with per-class AP) and DeepPCB's IoU>0.33 AP."""
    coco_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    coco_metric.update(predictions, targets)
    coco = coco_metric.compute()

    deeppcb_metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[config.DEEPPCB_IOU_THRESHOLD],
        class_metrics=True,
    )
    deeppcb_metric.update(predictions, targets)
    deeppcb = deeppcb_metric.compute()

    per_class = {}
    for index, class_id in enumerate(coco["classes"].tolist()):
        per_class[config.CLASS_NAMES[class_id]] = {
            "AP@[0.5:0.95]": float(coco["map_per_class"][index]),
            f"AP@{config.DEEPPCB_IOU_THRESHOLD}": float(deeppcb["map_per_class"][index]),
        }

    return {
        "mAP@[0.5:0.95]": float(coco["map"]),
        "mAP@0.5": float(coco["map_50"]),
        "mAP@0.75": float(coco["map_75"]),
        f"mAP@{config.DEEPPCB_IOU_THRESHOLD} (DeepPCB benchmark IoU)": float(deeppcb["map"]),
        "mAP_small": float(coco["map_small"]),
        "mAP_medium": float(coco["map_medium"]),
        "per_class": per_class,
    }


def plot_per_class_ap(metrics: Dict, output_path, split_name: str = "test") -> None:
    """Draw a horizontal bar chart of per-class AP, best class at the top.

    One measure across six categories, so the bars share a single hue -- colour
    would be encoding nothing that the axis does not already say -- and each bar
    is directly labelled with its value instead of relying on gridline reading.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_class = metrics["per_class"]
    ordered = sorted(per_class.items(), key=lambda kv: kv[1]["AP@[0.5:0.95]"])
    names = [name.replace("_", " ") for name, _ in ordered]
    values = [entry["AP@[0.5:0.95]"] for _, entry in ordered]

    fig, axes = plt.subplots(figsize=(8, 4.2))
    bars = axes.barh(names, values, color="#4363d8", height=0.5)
    for bar, value in zip(bars, values):
        axes.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color="#333333",
        )

    # The mean line is annotated at the bottom of the plot area: labelling it at
    # the top collides with the title, and mid-plot it collides with whichever
    # bar happens to sit near the mean.
    mean_ap = metrics["mAP@[0.5:0.95]"]
    axes.axvline(mean_ap, color="#888888", linestyle="--", linewidth=1)
    axes.text(
        mean_ap + 0.008,
        -0.62,
        f"mAP {mean_ap:.3f}",
        fontsize=8,
        color="#666666",
        va="center",
    )
    axes.set_ylim(-0.9, len(names) - 0.4)

    axes.set_xlim(0, min(1.0, max(values) * 1.18))
    axes.set_xlabel("AP@[0.5:0.95]")
    axes.set_title(f"Per-class average precision - DeepPCB {split_name} split", fontsize=11)
    axes.grid(axis="x", alpha=0.25, linewidth=0.6)
    axes.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axes.spines[spine].set_visible(False)
    axes.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def plot_predictions_vs_truth(model, dataset, device, output_path, num_images: int = 4) -> None:
    """Save a grid of test images: ground truth on the left, predictions right."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .visualize import draw_boxes

    model.eval()
    stride = max(1, len(dataset) // num_images)
    fig, axes = plt.subplots(num_images, 2, figsize=(11, 5.4 * num_images))

    for row in range(num_images):
        image, target = dataset[row * stride]
        with torch.no_grad():
            prediction = model([image.to(device)])[0]

        keep = prediction["scores"] >= config.SCORE_THRESHOLD
        draw_boxes(
            axes[row][0],
            image,
            target["boxes"],
            target["labels"],
            title=f"{target['image_id']} - ground truth ({len(target['boxes'])} boxes)",
        )
        draw_boxes(
            axes[row][1],
            image,
            prediction["boxes"][keep].cpu(),
            prediction["labels"][keep].cpu(),
            scores=prediction["scores"][keep].cpu(),
            title=(
                f"{target['image_id']} - predicted "
                f"({int(keep.sum())} boxes at score >= {config.SCORE_THRESHOLD})"
            ),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def print_report(metrics: Dict, split_name: str, num_images: int) -> None:
    """Print the headline metrics and the per-class breakdown."""
    print(f"\n=== Overall ({split_name} split, {num_images} images) ===")
    for key, value in metrics.items():
        if key != "per_class":
            print(f"  {key:<40} {value:.4f}")

    print("\n=== Per class ===")
    deeppcb_key = f"AP@{config.DEEPPCB_IOU_THRESHOLD}"
    print(f"  {'class':<18}{'AP@[0.5:0.95]':>15}{deeppcb_key:>14}")
    for name, entry in sorted(
        metrics["per_class"].items(), key=lambda kv: -kv[1]["AP@[0.5:0.95]"]
    ):
        print(f"  {name:<18}{entry['AP@[0.5:0.95]']:>15.4f}{entry[deeppcb_key]:>14.4f}")


def main() -> None:
    """Evaluate the checkpoint on the test set and write metrics and plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(config.MODELS_DIR / config.CHECKPOINT_NAME))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    device = config.get_device()
    model = load_checkpoint(args.checkpoint, device)
    print(f"loaded {args.checkpoint} on {device}")

    datasets = build_datasets()
    dataset = datasets[args.split]
    loader = build_dataloader(dataset, shuffle=False)
    print(f"evaluating on {args.split}: {len(dataset)} images")

    predictions, targets = collect_predictions(model, loader, device)
    metrics = compute_metrics(predictions, targets)
    print_report(metrics, args.split, len(dataset))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
    plot_per_class_ap(metrics, config.RESULTS_DIR / f"per_class_ap_{args.split}.png", args.split)
    plot_predictions_vs_truth(
        model, dataset, device, config.RESULTS_DIR / f"predictions_vs_truth_{args.split}.png"
    )
    print(f"\nwrote metrics and plots to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
