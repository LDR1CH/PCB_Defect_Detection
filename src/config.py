"""Central configuration for the PCB defect detector.

All filesystem paths, dataset conventions and hyperparameters live here so that
no function elsewhere in ``src/`` hardcodes a path or a magic number.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PCB_DATA_DIR = DATA_RAW_DIR / "DeepPCB" / "PCBData"

# Official DeepPCB split lists (1000 train/val images, 500 test images).
TRAINVAL_LIST = PCB_DATA_DIR / "trainval.txt"
TEST_LIST = PCB_DATA_DIR / "test.txt"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Dataset conventions ---------------------------------------------------
# Taken verbatim from the DeepPCB README: "type is an integer ID that follows
# the matches: 0-background (not used), 1-open, 2-short, 3-mousebite, 4-spur,
# 5-copper, 6-pin-hole".  The IDs already start at 1 and reserve 0 for
# background, which is exactly torchvision's Faster R-CNN convention, so the
# raw `type` values are used as labels directly -- NO +1 shift is applied.
CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "spurious_copper",  # DeepPCB's own eval script calls this "copper"
    6: "pin_hole",
}

BACKGROUND_LABEL = 0
NUM_CLASSES = len(CLASS_NAMES) + 1  # 6 defect classes + background = 7

# Distinct colours per class for drawing boxes.  Checked with the dataviz
# palette validator: lightness band, chroma floor and normal-vision separation
# all pass; the worst colourblind pair (open/red vs short/green) sits at deutan
# dE 7.2, which is legal only alongside a secondary encoding -- every box is
# drawn with its class name as a text label, which supplies that.  Contrast vs
# a white background is likewise carried by those labels.
CLASS_COLORS = {
    1: "#e6194b",
    2: "#3cb44b",
    3: "#4363d8",
    4: "#f58231",
    5: "#911eb4",
    6: "#00b8b8",
}

# The tested images in DeepPCB are the defective ones; each is paired with a
# defect-free "_temp" template image that this project does not use.
TEST_IMAGE_SUFFIX = "_test"
TEMPLATE_IMAGE_SUFFIX = "_temp"

IMAGE_SIZE = (640, 640)  # (width, height), constant across the dataset

# --- Splits and loading ----------------------------------------------------
# Validation is carved out of the official trainval list at the IMAGE level
# with a fixed seed, so the split is reproducible across runs and sessions.
#
# Caveat worth remembering when reading val numbers: the official test set is
# dominated by board groups that never appear in trainval (332 of its 500
# images come from group12000/12100/12300/90100), whereas an image-level val
# split shares boards with train.  Val is therefore an optimistic progress
# signal for tuning; the official test set is the honest generalization number.
SEED = 42
VAL_FRACTION = 0.1

# Peak VRAM measured at 1.94 GB for a 2-image batch on a 16 GB card, so 8 fits
# comfortably; lower this first if training hits an out-of-memory error.
BATCH_SIZE = 8
NUM_WORKERS = 4

# --- Training hyperparameters ----------------------------------------------
# SGD settings are the torchvision detection reference values;
# the step schedule decays LR once the loss curve flattens.
EPOCHS = 10
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 3   # epochs between LR decays
LR_GAMMA = 0.1
PRINT_EVERY = 20   # iterations between progress lines

CHECKPOINT_NAME = "fasterrcnn_resnet50_v1.pth"

# --- Inference / evaluation ------------------------------------------------
# Boxes below this score are not drawn (mAP itself is computed on the full,
# unthresholded prediction set -- thresholding first would distort recall).
SCORE_THRESHOLD = 0.5
# DeepPCB's own benchmark counts a detection correct at IoU > 0.33, so that
# threshold is reported alongside the standard COCO ones for comparability.
DEEPPCB_IOU_THRESHOLD = 0.33


# --- Compute ---------------------------------------------------------------
def get_device():
    """Return the training/inference device, preferring CUDA when available.

    Guard against a silently CPU-only PyTorch build: on Windows a plain
    `pip install torch` pulls the CPU wheel, which makes training ~30x slower.
    See requirements.txt for the CUDA index this project installs from.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    print(
        f"WARNING: CUDA unavailable (torch {torch.__version__}, "
        f"built for CUDA {torch.version.cuda}) - falling back to CPU."
    )
    return torch.device("cpu")
