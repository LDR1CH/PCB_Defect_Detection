"""PyTorch Dataset, splits, augmentation and collate_fn for DeepPCB.

The Dataset returns the FULL board image with every defect box in it -- this is
a detection task, so nothing is cropped out.  Run ``python -m src.dataset`` for
a self-check that prints the split sizes and one decoded sample, and writes an
augmented batch visualization to ``results/data_checks/``.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from . import config
from .data_utils import Sample, load_split, parse_annotation_file

Target = Dict[str, torch.Tensor]


class DeepPCBDataset(Dataset):
    """Full tested board images paired with all of their defect boxes.

    ``__getitem__`` returns ``(image, target)`` where `image` is a float32
    CHW tensor scaled to [0, 1] and `target` holds `boxes` (FloatTensor[N, 4]
    in absolute ``[x1, y1, x2, y2]`` pixels) and `labels` (Int64Tensor[N] with
    DeepPCB's own 1-6 class IDs, which already reserve 0 for background).
    """

    def __init__(self, samples: Sequence[Sample], transforms: Optional[v2.Transform] = None):
        """Store the sample list and the (optional) joint image+box transform."""
        self.samples = list(samples)
        self.transforms = transforms

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Target]:
        """Load one image and build its detection target."""
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")  # 3ch: the COCO-pretrained
        # backbone expects RGB, though DeepPCB images are binarized grayscale.
        defects = parse_annotation_file(sample.annotation_path)

        boxes = torch.tensor([d[:4] for d in defects], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([d[4] for d in defects], dtype=torch.int64)

        target: Target = {
            # BoundingBoxes is a tv_tensor, so v2 transforms move the boxes
            # with the image instead of silently leaving them behind.
            "boxes": tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=(image.height, image.width)
            ),
            "labels": labels,
            # Plain int (not a tensor): v2 passes unknown types through, but a
            # bare tensor in the dict risks being mistaken for image data.
            "image_id": int(sample.image_id),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def get_transforms(train: bool) -> v2.Transform:
    """Build the joint image+box transform pipeline for a split.

    Only horizontal flip is used for now (kept
    simple initially); vertical flip is equally valid for PCB images (a board
    has no canonical up) and is the obvious next addition if the baseline
    underfits.
    """
    transforms: List[v2.Transform] = [v2.ToImage()]
    if train:
        transforms.append(v2.RandomHorizontalFlip(p=0.5))
    transforms.append(v2.ToDtype(torch.float32, scale=True))
    # Drops any box left degenerate or outside the canvas by a transform.
    transforms.append(v2.SanitizeBoundingBoxes())
    return v2.Compose(transforms)


def collate_fn(batch: Sequence[Tuple[torch.Tensor, Target]]) -> Tuple[Tuple, Tuple]:
    """Batch samples without stacking, since images hold differing box counts.

    torchvision detection models take a *list* of images and a *list* of target
    dicts, so the default collate (which would try to stack the [N, 4] box
    tensors) cannot be used.
    """
    return tuple(zip(*batch))


def make_splits() -> Dict[str, List[Sample]]:
    """Split the data into train/val/test at the image level.

    Test is DeepPCB's official 500-image list, untouched.  Val is a fixed
    random `config.VAL_FRACTION` of the official trainval list; the seed makes
    this reproducible across sessions.
    """
    trainval = load_split(config.TRAINVAL_LIST)
    test = load_split(config.TEST_LIST)

    shuffled = list(trainval)
    random.Random(config.SEED).shuffle(shuffled)
    num_val = int(len(shuffled) * config.VAL_FRACTION)
    return {"train": shuffled[num_val:], "val": shuffled[:num_val], "test": test}


def build_datasets() -> Dict[str, DeepPCBDataset]:
    """Return the train/val/test Datasets, with augmentation on train only."""
    splits = make_splits()
    return {
        name: DeepPCBDataset(samples, get_transforms(train=(name == "train")))
        for name, samples in splits.items()
    }


def build_dataloader(dataset: DeepPCBDataset, shuffle: bool, **kwargs) -> DataLoader:
    """Wrap a Dataset in a DataLoader using this project's collate_fn."""
    params = {
        "batch_size": config.BATCH_SIZE,
        "num_workers": config.NUM_WORKERS,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    params.update(kwargs)
    # Windows spawns (rather than forks) workers, so re-creating them every
    # epoch costs seconds per epoch; keep them alive between epochs instead.
    params["persistent_workers"] = params["num_workers"] > 0
    return DataLoader(dataset, shuffle=shuffle, **params)


def _check_flip_moves_boxes(samples: Sequence[Sample]) -> None:
    """Assert a horizontal flip mirrors the boxes, not just the pixels.

    This is the failure mode detection augmentation is notorious for: the image
    flips, the boxes stay put, and the model trains on silently wrong targets.
    """
    plain = DeepPCBDataset(samples[:1], get_transforms(train=False))
    always_flip = DeepPCBDataset(
        samples[:1],
        v2.Compose(
            [v2.ToImage(), v2.RandomHorizontalFlip(p=1.0), v2.ToDtype(torch.float32, scale=True)]
        ),
    )
    image, target = plain[0]
    flipped_image, flipped_target = always_flip[0]

    width = image.shape[-1]
    expected_x1 = width - target["boxes"][:, 2]
    expected_x2 = width - target["boxes"][:, 0]
    assert torch.allclose(flipped_target["boxes"][:, 0], expected_x1, atol=1e-4)
    assert torch.allclose(flipped_target["boxes"][:, 2], expected_x2, atol=1e-4)
    assert torch.equal(flipped_target["boxes"][:, 1], target["boxes"][:, 1])  # y unchanged
    assert torch.equal(flipped_image, torch.flip(image, dims=[-1]))
    assert torch.equal(flipped_target["labels"], target["labels"])
    print("  flip check: boxes mirror with the image (x1'=W-x2, y unchanged), labels intact")


def main() -> None:
    """Verify the splits, one decoded sample, a collated batch and augmentation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .visualize import draw_boxes

    print("=== splits (image level) ===")
    splits = make_splits()
    for name, samples in splits.items():
        print(f"  {name:<5} {len(samples):>4} images")
    ids = {name: {s.image_id for s in samples} for name, samples in splits.items()}
    assert not ids["train"] & ids["val"], "train/val overlap"
    assert not (ids["train"] | ids["val"]) & ids["test"], "trainval/test overlap"
    print(f"  no image id appears in two splits ({sum(map(len, ids.values()))} unique ids)")

    datasets = build_datasets()
    image, target = datasets["train"][0]
    print("\n=== one sample from the train split ===")
    print(f"  image  {tuple(image.shape)} {image.dtype} range [{image.min():.2f}, {image.max():.2f}]")
    print(f"  boxes  {tuple(target['boxes'].shape)} {target['boxes'].dtype}")
    print(f"  labels {tuple(target['labels'].shape)} {target['labels'].dtype} -> {target['labels'].tolist()}")
    print(f"  image_id {target['image_id']}")
    assert image.dtype == torch.float32 and image.shape[0] == 3
    assert target["boxes"].shape[0] == target["labels"].shape[0]
    assert target["labels"].min() >= 1 and target["labels"].max() <= len(config.CLASS_NAMES)

    print("\n=== augmentation ===")
    _check_flip_moves_boxes(splits["train"])

    print("\n=== collate_fn on a real DataLoader batch ===")
    # num_workers=0 here: worker processes add startup cost for a one-batch check.
    loader = build_dataloader(datasets["train"], shuffle=True, num_workers=0)
    images, targets = next(iter(loader))
    print(f"  batch of {len(images)} images, {len(targets)} targets (tuples, not stacked)")
    print(f"  boxes per image in batch: {[len(t['boxes']) for t in targets]}")
    assert len(images) == len(targets) == config.BATCH_SIZE
    assert all(img.shape == images[0].shape for img in images)

    output_dir = config.RESULTS_DIR / "data_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, img, tgt in zip(axes.ravel(), images, targets):
        draw_boxes(
            ax,
            img,
            tgt["boxes"],
            tgt["labels"],
            title=f"{tgt['image_id']} - {len(tgt['boxes'])} boxes (augmented)",
        )
    fig.tight_layout()
    output_path = output_dir / "train_batch_augmented.png"
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {output_path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
