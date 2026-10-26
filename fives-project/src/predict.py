"""Single-image inference: raw fundus image (path or RGB array) -> (label_name, {class_name: probability}).

Runs exactly the training path: resize -> FOV crop + circular mask -> (illumination/channel normalisation for the
`handcrafted_norm` feature set) -> handcrafted features (and a ResNet50 embedding of the preprocessed image if the saved
model uses it) -> saved model. What the model expects is read from models/best_model_meta.json.
Raises ValueError if no fundus field of view is found (e.g. a blank or non-fundus image).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import features as F  # noqa: E402

_STATE = {}


def _load():
    if not _STATE:
        _STATE["model"] = joblib.load(config.MODELS_DIR / "best_model.joblib")
        _STATE["meta"] = json.loads((config.MODELS_DIR / "best_model_meta.json").read_text())
        _STATE["classes"] = {int(k): v for k, v in json.loads((config.MODELS_DIR / "classes.json").read_text()).items()}
    return _STATE


def _resnet_embedding(img):
    """ResNet50 (ImageNet) global-average-pooled 2048-d feature of the preprocessed image, as in scripts/04_extract_features.py."""
    import torch
    import torchvision

    if "resnet" not in _STATE:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        net = net.eval().to(dev)
        _STATE["resnet"] = (net.half() if dev == "cuda" else net, dev)
    net, dev = _STATE["resnet"]
    x = torch.from_numpy(F.resnet_input(img)).permute(2, 0, 1)[None].float().div(255).to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        return net(x.half() if dev == "cuda" else x).float().cpu().numpy()[0]


def extract_features(path_or_array):
    """One-row DataFrame with the columns (and order) the saved model was trained on."""
    st = _load()
    fs = st["meta"]["feature_set"]
    img = F.as_rgb(path_or_array)
    row = {}
    if fs == "handcrafted_norm":
        row.update(F.extract_handcrafted_norm(img))
    if fs in ("handcrafted", "both"):
        row.update(F.extract_handcrafted(img))
    if fs in ("resnet", "both"):
        row.update({f"resnet_{i}": v for i, v in enumerate(_resnet_embedding(img))})
    return pd.DataFrame([row])[st["meta"]["feature_columns"]]


def predict_image(path_or_array):
    """-> (label_name, {class_name: probability}) for one raw fundus image (file path or HxWx3 uint8 RGB array)."""
    st = _load()
    proba = st["model"].predict_proba(extract_features(path_or_array))[0]
    probs = {st["classes"][int(c)]: float(p) for c, p in zip(st["model"].classes_, proba)}
    return max(probs, key=probs.get), probs
