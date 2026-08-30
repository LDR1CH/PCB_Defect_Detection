"""Helpers for reading the DeepPCB split lists and annotation files.

Shared by the Phase 1 inspection script and (later) the PyTorch Dataset, so the
quirks of the on-disk layout are handled in exactly one place.
"""

from pathlib import Path
from typing import List, NamedTuple, Tuple

from . import config


class Sample(NamedTuple):
    """One tested image and its annotation file."""

    image_path: Path
    annotation_path: Path

    @property
    def image_id(self) -> str:
        """Return the DeepPCB image id, e.g. '20085000'."""
        return self.annotation_path.stem


def resolve_image_path(listed_path: Path) -> Path:
    """Resolve an image path as written in a split list to the real file.

    The split lists name images as ``<group>/<id>/<id>.jpg``, but on disk the
    tested image is ``<id>_test.jpg`` (its defect-free partner being
    ``<id>_temp.jpg``).  Fall back to inserting the ``_test`` suffix when the
    listed name does not exist.
    """
    if listed_path.exists():
        return listed_path
    with_suffix = listed_path.with_name(
        f"{listed_path.stem}{config.TEST_IMAGE_SUFFIX}{listed_path.suffix}"
    )
    if with_suffix.exists():
        return with_suffix
    raise FileNotFoundError(f"No image found for {listed_path} (tried {with_suffix})")


def load_split(list_path: Path) -> List[Sample]:
    """Read a DeepPCB split list into `Sample` pairs of resolved absolute paths.

    Each line is ``<image path> <annotation path>``, both relative to PCBData/.
    """
    samples = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        image_rel, annotation_rel = line.split()
        image_path = resolve_image_path(config.PCB_DATA_DIR / image_rel)
        annotation_path = config.PCB_DATA_DIR / annotation_rel
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        samples.append(Sample(image_path, annotation_path))
    return samples


def parse_annotation_file(annotation_path: Path) -> List[Tuple[int, int, int, int, int]]:
    """Parse one DeepPCB annotation file into `(x1, y1, x2, y2, type)` tuples.

    Coordinates are absolute pixels; `type` is the raw DeepPCB class ID (1-6).
    """
    defects = []
    for line_no, line in enumerate(annotation_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 5:
            raise ValueError(
                f"{annotation_path}:{line_no}: expected 5 values, got {len(parts)}: {line!r}"
            )
        x1, y1, x2, y2, defect_type = (int(p) for p in parts)
        defects.append((x1, y1, x2, y2, defect_type))
    return defects
