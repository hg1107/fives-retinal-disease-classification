"""04 - Extract features for every image in manifest.csv.

Outputs (data/processed/): features_handcrafted.csv, features_resnet.npy + features_resnet_ids.csv.
Handcrafted features are cached per image (data/processed/cache/hand_<version>/<image_id>.json) so a
crashed run resumes. Choice: ResNet50 is fed the preprocessed image (FOV-cropped and FOV-masked, no CLAHE),
not the raw 2048 px frame, because EDA showed the raw frame carries a corner glyph that marks the image
source (3% of train Normal vs ~86% of test Normal) and a CNN would exploit it. Masks are never read.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import features as F  # noqa: E402

N_JOBS = 8  # 16 cores, but each worker holds several 2048x2048 arrays; 8 keeps RAM comfortable
BATCH = 32  # ResNet50 fp16 at 224 px needs well under 2 GB, safe for a 4 GB card

m = pd.read_csv(config.DATA_DIR / "manifest.csv")
D = config.DATASET_PATH


# ---------------------------------------------------------------- handcrafted (base + normalised variant)
def one(image_id, rel, extractor, cache):
    f = cache / f"{image_id}.json"
    if f.exists():
        return image_id, json.loads(f.read_text())
    try:
        feats = extractor(D / rel)
    except Exception as e:
        raise RuntimeError(f"{image_id}: {type(e).__name__}: {e}") from e
    f.write_text(json.dumps(feats))
    return image_id, feats


def handcrafted(extractor, version, out_name):
    cache = config.DATA_DIR / "cache" / f"hand_{version}"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"\n{out_name}: {len(m)} images ({sum((cache / f'{i}.json').exists() for i in m.image_id)} already cached)")
    res = dict(tqdm(Parallel(n_jobs=N_JOBS, return_as="generator")(
        delayed(one)(i, p, extractor, cache) for i, p in zip(m.image_id, m.image_path)), total=len(m), desc=out_name))
    H = pd.DataFrame([res[i] for i in m.image_id])
    H.insert(0, "split", m.official_split.values)
    H.insert(0, "label_id", m.label_id.values)
    H.insert(0, "image_id", m.image_id.values)
    H.to_csv(config.DATA_DIR / out_name, index=False)
    print(f"Saved {out_name}: {H.shape[0]} rows, {H.shape[1] - 3} features")
    print(pd.Series([c.split("_")[0] for c in H.columns[3:]]).value_counts().to_string())
    return H


H = handcrafted(F.extract_handcrafted, F.FEATURE_VERSION, "features_handcrafted.csv")
HN = handcrafted(F.extract_handcrafted_norm, F.FEATURE_VERSION_NORM, "features_handcrafted_norm.csv")
fcols = H.columns[3:]


# ---------------------------------------------------------------- ResNet50 embeddings
class Imgs(torch.utils.data.Dataset):
    def __len__(self):
        return len(m)

    def __getitem__(self, i):
        return torch.from_numpy(F.resnet_input(D / m.image_path[i])).permute(2, 0, 1)  # uint8 CHW


def embed():
    assert torch.cuda.is_available(), "CUDA not available"
    dev = "cuda"
    net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()  # global-average-pooled 2048-d features
    net = net.eval().half().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    dl = torch.utils.data.DataLoader(Imgs(), batch_size=BATCH, shuffle=False, num_workers=N_JOBS)
    out = []
    with torch.no_grad():
        for x in tqdm(dl, desc="resnet50"):
            x = ((x.to(dev).float() / 255 - mean) / std).half()
            out.append(net(x).float().cpu().numpy())
    E = np.concatenate(out).astype(np.float32)
    np.save(config.DATA_DIR / "features_resnet.npy", E)
    m[["image_id", "label_id", "official_split"]].rename(columns={"official_split": "split"}).to_csv(ids_path, index=False)
    print(f"Saved features_resnet.npy {E.shape}; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return E


ids_path = config.DATA_DIR / "features_resnet_ids.csv"
if (config.DATA_DIR / "features_resnet.npy").exists() and ids_path.exists() and (pd.read_csv(ids_path).image_id.values == m.image_id.values).all():
    print("\nResNet embeddings already extracted for this manifest; skipping (delete features_resnet.npy to redo).")
    E = np.load(config.DATA_DIR / "features_resnet.npy")
else:
    E = embed()

# ---------------------------------------------------------------- NaN / inf report
def report(name, a, cols):
    a = np.asarray(a, dtype=np.float64)
    nan, inf = np.isnan(a), np.isinf(a)
    print(f"{name}: NaN={nan.sum()} inf={inf.sum()} (rows affected {(nan | inf).any(1).sum()})")
    for j in np.where((nan | inf).any(0))[0]:
        print(f"   {cols[j]}: {nan[:, j].sum()} NaN, {inf[:, j].sum()} inf")
    const = np.where(np.nanstd(a, axis=0) == 0)[0]
    if len(const):
        print(f"   constant columns: {[cols[j] for j in const]}")


print("\nValidity checks")
report("handcrafted", H[fcols].values, list(fcols))
report("handcrafted_norm", HN[fcols].values, list(fcols))
report("resnet", E, [f"r{j}" for j in range(E.shape[1])])
assert (H.image_id.values == m.image_id.values).all() and (HN.image_id.values == m.image_id.values).all()
