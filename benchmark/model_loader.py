"""Load benchmark checkpoints without downloading initialization weights."""
from pathlib import Path
from functools import partial
from train_cv import CLASS_NAMES

def build_torchvision_model(model_name: str, checkpoint: Path, device):
    import torch
    import torch.nn as nn
    from torchvision.models.detection import (
        fcos_resnet50_fpn,
        fasterrcnn_resnet50_fpn_v2,
        retinanet_resnet50_fpn_v2,
    )
    from torchvision.models.detection.fcos import FCOSClassificationHead
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    if model_name == "fasterrcnn_r50_fpn_v2":
        model = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            min_size=640,
            max_size=640,
            box_score_thresh=0.001,
            box_nms_thresh=0.70,
            box_detections_per_img=100,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
    elif model_name == "retinanet_r50_fpn_v2":
        model = retinanet_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            min_size=640,
            max_size=640,
            score_thresh=0.001,
            nms_thresh=0.70,
            detections_per_img=100,
        )
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels,
            model.head.classification_head.num_anchors,
            len(CLASS_NAMES) + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
    elif model_name == "fcos_r50_fpn":
        model = fcos_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            min_size=640,
            max_size=640,
            score_thresh=0.001,
            nms_thresh=0.70,
            detections_per_img=100,
        )
        model.head.classification_head = FCOSClassificationHead(
            model.backbone.out_channels,
            model.head.classification_head.num_anchors,
            len(CLASS_NAMES) + 1,
            norm_layer=partial(nn.GroupNorm, 32),
        )
    else:
        raise ValueError(model_name)

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
