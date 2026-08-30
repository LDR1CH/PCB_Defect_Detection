"""Gradio web demo for the PCB defect detector.

Upload a board image, get it back annotated with predicted defect boxes, class
names and confidence scores.

Run from the repo root:  python app/demo.py
"""

import argparse
import sys
from pathlib import Path

# Allow `python app/demo.py` (not just `python -m app.demo`) by putting the
# repo root on the path before importing the project package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms import v2

from src import config
from src.model import load_checkpoint
from src.visualize import draw_boxes_pil

TO_TENSOR = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

DESCRIPTION = """
Upload a 640x640 DeepPCB-style board image (binarized, tested image) to locate
and classify its defects: **open, short, mousebite, spur, spurious copper,
pin hole**.

Faster R-CNN (ResNet50-FPN) fine-tuned on DeepPCB — **mAP@0.5 = 0.961**,
mAP@[0.5:0.95] = 0.745 on the official 500-image test set.
"""


def predict(image: Image.Image, score_threshold: float, model, device):
    """Detect defects in `image`, returning the annotated image and a summary.

    Boxes scoring below `score_threshold` are dropped, since an unfiltered
    Faster R-CNN returns up to 100 low-confidence boxes per image.
    """
    if image is None:
        return None, "Upload an image to run detection."

    tensor = TO_TENSOR(image.convert("RGB")).to(device)
    with torch.no_grad():
        prediction = model([tensor])[0]

    keep = prediction["scores"] >= score_threshold
    boxes = prediction["boxes"][keep].cpu()
    labels = prediction["labels"][keep].cpu()
    scores = prediction["scores"][keep].cpu()

    annotated = draw_boxes_pil(image, boxes, labels, scores)

    if len(boxes) == 0:
        return annotated, f"No defects found at score >= {score_threshold:.2f}."

    counts: dict = {}
    for label in labels.tolist():
        name = config.CLASS_NAMES[label]
        counts[name] = counts.get(name, 0) + 1
    breakdown = ", ".join(f"{count} x {name}" for name, count in sorted(counts.items()))
    return annotated, f"Found {len(boxes)} defect(s): {breakdown}."


def find_example_images(limit: int = 4):
    """Return a few test-split images to prefill the demo's examples, if present.

    The dataset is not committed, so this quietly returns nothing when the demo
    runs on a machine without `data/raw/` populated.
    """
    try:
        from src.data_utils import load_split

        samples = load_split(config.TEST_LIST)
    except (FileNotFoundError, OSError):
        return []
    stride = max(1, len(samples) // limit)
    return [[str(samples[i * stride].image_path)] for i in range(limit)]


def build_interface(model, device) -> gr.Blocks:
    """Assemble the Gradio UI around the loaded model."""
    with gr.Blocks(title="PCB Defect Detector") as demo:
        gr.Markdown("# PCB Defect Detector")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Board image")
                threshold = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=config.SCORE_THRESHOLD,
                    step=0.05,
                    label="Confidence threshold",
                )
                detect_button = gr.Button("Detect defects", variant="primary")
            with gr.Column():
                image_output = gr.Image(type="pil", label="Detected defects")
                summary = gr.Textbox(label="Summary", lines=2)

        examples = find_example_images()
        if examples:
            gr.Examples(examples=examples, inputs=image_input, label="Test-set examples")

        detect_button.click(
            fn=lambda image, threshold_value: predict(image, threshold_value, model, device),
            inputs=[image_input, threshold],
            outputs=[image_output, summary],
        )

    return demo


def main() -> None:
    """Load the checkpoint and launch the Gradio server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(config.MODELS_DIR / config.CHECKPOINT_NAME))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"No checkpoint at {checkpoint_path}. Train one first: python -m src.train"
        )

    device = config.get_device()
    model = load_checkpoint(checkpoint_path, device)
    print(f"loaded {checkpoint_path.name} on {device}")

    build_interface(model, device).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
