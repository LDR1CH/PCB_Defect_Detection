"""Faster R-CNN model construction for PCB defect detection."""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import config


def build_model(pretrained: bool = True) -> torch.nn.Module:
    """Build `fasterrcnn_resnet50_fpn` with a box predictor sized for this task.

    The COCO-pretrained head predicts 91 classes; it is replaced with a fresh
    `FastRCNNPredictor` over `config.NUM_CLASSES` (6 defects + background).
    Everything else -- backbone, FPN, RPN -- keeps its pretrained weights.
    """
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.NUM_CLASSES)
    return model


def load_checkpoint(checkpoint_path, device=None) -> torch.nn.Module:
    """Rebuild the architecture and load trained weights from `checkpoint_path`."""
    device = device or config.get_device()
    model = build_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
