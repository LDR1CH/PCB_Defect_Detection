"""Manual training loop for the Faster R-CNN PCB defect detector.

torchvision detection models return a dict of losses in ``train()`` mode, so
there is no separate loss function: the four losses (classifier, box
regression, RPN objectness, RPN box regression) are summed and backpropagated.

Run from the repo root:  python -m src.train --epochs 10
"""

import argparse
import json
import time
from itertools import islice
from typing import Dict, List

import torch

from . import config
from .dataset import build_dataloader, build_datasets
from .model import build_model


class LimitedLoader:
    """Wrap a DataLoader to yield only its first `num_batches` batches.

    Used by ``--limit-batches`` so the whole pipeline can be smoke-tested in
    seconds without waiting for full epochs.
    """

    def __init__(self, loader, num_batches: int):
        """Store the wrapped loader and the batch cap."""
        self.loader = loader
        self.num_batches = num_batches

    def __iter__(self):
        """Yield at most `num_batches` batches from the wrapped loader."""
        return islice(iter(self.loader), self.num_batches)

    def __len__(self) -> int:
        """Return the capped batch count."""
        return min(self.num_batches, len(self.loader))


def run_epoch(model, loader, device, optimizer=None) -> Dict[str, float]:
    """Run one pass over `loader`, returning mean total and per-component loss.

    With `optimizer` given this trains; without one it evaluates.  The model is
    kept in ``train()`` mode either way, because that is the only mode in which
    torchvision returns losses rather than predictions -- safe here because the
    pretrained backbone uses FrozenBatchNorm, so a validation pass updates no
    running statistics.
    """
    is_training = optimizer is not None
    model.train()
    totals: Dict[str, float] = {}
    num_batches = 0

    for iteration, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in target.items()}
            for target in targets
        ]

        with torch.set_grad_enabled(is_training):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at iteration {iteration}: {loss_dict}")

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        totals["loss"] = totals.get("loss", 0.0) + loss.item()
        for name, value in loss_dict.items():
            totals[name] = totals.get(name, 0.0) + value.item()
        num_batches += 1

        if is_training and iteration % config.PRINT_EVERY == 0:
            print(f"    iter {iteration:>4}/{len(loader)}  loss {loss.item():.4f}", flush=True)

    return {name: total / num_batches for name, total in totals.items()}


def save_loss_curve(history: List[Dict], output_path) -> None:
    """Plot total loss and per-component training loss per epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in history]
    fig, (ax_total, ax_parts) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_total.plot(epochs, [e["train"]["loss"] for e in history], marker="o", label="train")
    ax_total.plot(epochs, [e["val"]["loss"] for e in history], marker="o", label="val")
    ax_total.set(xlabel="epoch", ylabel="total loss", title="Total loss")
    ax_total.legend()
    ax_total.grid(alpha=0.3)

    for name in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
        ax_parts.plot(epochs, [e["train"][name] for e in history], marker=".", label=name)
    ax_parts.set(xlabel="epoch", ylabel="loss", title="Training loss components")
    ax_parts.legend(fontsize=8)
    ax_parts.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Train the detector, saving the best-val checkpoint and a loss curve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="stop each epoch after N batches (smoke test only)",
    )
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    device = config.get_device()
    print(f"device: {device} | epochs: {args.epochs} | batch: {args.batch_size} | lr: {args.lr}")

    datasets = build_datasets()
    train_loader = build_dataloader(datasets["train"], shuffle=True, batch_size=args.batch_size)
    val_loader = build_dataloader(datasets["val"], shuffle=False, batch_size=args.batch_size)
    print(
        f"train {len(datasets['train'])} images / {len(train_loader)} batches, "
        f"val {len(datasets['val'])} images / {len(val_loader)} batches"
    )

    if args.limit_batches:
        train_loader = LimitedLoader(train_loader, args.limit_batches)
        val_loader = LimitedLoader(val_loader, args.limit_batches)

    model = build_model(pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.MODELS_DIR / config.CHECKPOINT_NAME

    history: List[Dict] = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nepoch {epoch}/{args.epochs}  (lr {current_lr:g})", flush=True)

        train_losses = run_epoch(model, train_loader, device, optimizer)
        val_losses = run_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.perf_counter() - started

        print(
            f"  train loss {train_losses['loss']:.4f} | "
            f"val loss {val_losses['loss']:.4f} | {elapsed:.0f}s"
        )
        components = "  ".join(
            f"{name.replace('loss_', '')} {value:.4f}"
            for name, value in train_losses.items()
            if name != "loss"
        )
        print(f"  {components}")
        history.append(
            {"epoch": epoch, "train": train_losses, "val": val_losses, "seconds": elapsed}
        )

        if val_losses["loss"] < best_val:
            best_val = val_losses["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "num_classes": config.NUM_CLASSES,
                    "class_names": config.CLASS_NAMES,
                },
                checkpoint_path,
            )
            print(f"  new best val loss -> saved {checkpoint_path.name}")

    (config.RESULTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    save_loss_curve(history, config.RESULTS_DIR / "loss_curve.png")

    first, last = history[0]["train"]["loss"], history[-1]["train"]["loss"]
    verdict = "decreased" if last < first else "DID NOT DECREASE"
    print(f"\ntrain loss {first:.4f} -> {last:.4f} ({verdict})")
    print(f"best val loss {best_val:.4f} | checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
