"""Loaders for the four frozen feature extractors.

Each loader returns ``(forward_fn, preprocess)`` where ``forward_fn`` maps a
batched image tensor to an embedding, and ``preprocess`` is the torchvision /
open_clip transform for that model.

Embedding dimensionalities: resnet_baseline 2048, dermlip 512, monet 1024,
dinov3 768.

NOTE ON PREPROCESSING. To reproduce the paper exactly, keep the transforms
below as they are. They squash-resize to 224x224 rather than
resize-shortest-side + center-crop. That is a mild deviation from the official
CLIP/DINOv3 preprocessing and is documented as a limitation; passing
``--official-preprocess`` to the extraction script switches MONET and DINOv3 to
their canonical transforms for a robustness check.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGENET_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

CLIP_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

# Canonical resize-then-center-crop variants (robustness check only).
OFFICIAL_CLIP_TF = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

OFFICIAL_IMAGENET_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Training-time augmentation for the cancer baseline.
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

DERMLIP_HUB = "hf-hub:redlessone/DermLIP_ViT-B-16"
MONET_HUB = "chanwkim/monet"
DINOV3_HUB = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def build_baseline_resnet(n_classes: int = 8, pretrained: bool = True) -> nn.Module:
    """ResNet-50 with a fresh n-way head, for fine-tuning."""
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tvm.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def load_resnet_baseline(device, checkpoint: str, official: bool = False):
    model = build_baseline_resnet(pretrained=False)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.fc = nn.Identity()          # 2048-d penultimate features
    model = model.to(device).eval()
    tf = OFFICIAL_IMAGENET_TF if official else IMAGENET_TF
    return (lambda x: model(x)), tf


def load_dermlip(device, checkpoint=None, official: bool = False):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(DERMLIP_HUB)
    model = model.to(device).eval()
    # open_clip already supplies the model's own canonical transform.
    return (lambda x: model.encode_image(x)), preprocess


def load_monet(device, checkpoint=None, official: bool = False):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MONET_HUB).to(device).eval()
    tf = OFFICIAL_CLIP_TF if official else CLIP_TF
    return (lambda x: model.vision_model(pixel_values=x).pooler_output), tf


def load_dinov3(device, checkpoint=None, official: bool = False):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(DINOV3_HUB).to(device).eval()
    tf = OFFICIAL_IMAGENET_TF if official else IMAGENET_TF
    return (lambda x: model(pixel_values=x).pooler_output), tf


MODEL_LOADERS = {
    "resnet_baseline": load_resnet_baseline,
    "dermlip": load_dermlip,
    "monet": load_monet,
    "dinov3": load_dinov3,
}

from .config import EMBED_DIMS  # noqa: F401  (re-exported for convenience)
