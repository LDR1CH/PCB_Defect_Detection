"""Shared box-drawing helpers.

Used by the data sanity checks now, and by evaluation and the demo later, so
predicted and ground-truth boxes are always drawn the same way.
"""

from typing import Optional, Sequence

import matplotlib.patches as patches
import numpy as np

from . import config


def draw_boxes(
    axes,
    image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
) -> None:
    """Draw `boxes` with class-coloured outlines and labels onto `axes`.

    `image` may be a PIL image, an HWC array, or a CHW float tensor.  `scores`,
    when given, is appended to each label as a confidence percentage.
    """
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[0] != array.shape[-1]:
        array = array.transpose(1, 2, 0)  # CHW tensor -> HWC for imshow
    axes.imshow(array, cmap="gray" if array.ndim == 2 else None)

    for index, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = (float(v) for v in box)
        label = int(label)
        color = config.CLASS_COLORS.get(label, "white")
        axes.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor="none"
            )
        )
        text = f"{label}:{config.CLASS_NAMES.get(label, '?')}"
        if scores is not None:
            text += f" {float(scores[index]):.0%}"
        axes.text(
            x1,
            y1 - 4,
            text,
            color="white",
            fontsize=7,
            bbox=dict(facecolor=color, edgecolor="none", pad=1),
        )

    if title:
        axes.set_title(title, fontsize=9)
    axes.axis("off")


def draw_boxes_pil(image, boxes, labels, scores=None, line_width: int = 3):
    """Return a copy of `image` with boxes, class names and scores drawn on it.

    A PIL counterpart to `draw_boxes` for the web demo, which needs a plain
    image back rather than a matplotlib figure.  Colours match the class
    palette used everywhere else.
    """
    from PIL import Image, ImageDraw, ImageFont

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:  # font availability varies by platform; fall back silently
        font = ImageFont.load_default()

    for index, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = (float(v) for v in box)
        label = int(label)
        color = config.CLASS_COLORS.get(label, "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        text = config.CLASS_NAMES.get(label, str(label))
        if scores is not None:
            text += f" {float(scores[index]):.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = right - left, bottom - top
        # Keep the caption on-canvas when the box sits against the top edge.
        text_y = y1 - text_h - 4 if y1 - text_h - 4 >= 0 else y2 + 2
        draw.rectangle([x1, text_y, x1 + text_w + 6, text_y + text_h + 4], fill=color)
        draw.text((x1 + 3, text_y + 2), text, fill="white", font=font)

    return annotated
