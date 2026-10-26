"""01 - Audit the raw dataset: layout, labels, integrity, duplicates, image quality.

Layout is discovered from the folder tree (not assumed): a directory is treated as
"original" / "mask" by keywords in its path, and as train/test/val by keywords too.
Writes results/tables/audit_*.csv and results/audit_summary.md.
"""
import hashlib
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import imagehash  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ROOT = config.DATASET_PATH
RESULTS = config.ROOT / "results"
T = config.TABLES_DIR
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PHASH_NEAR = 6  # Hamming distance (of 64 bits) at or below which a pair is a near-duplicate CANDIDATE
CORR_CONFIRM = 0.99  # phash is too coarse for dark-background fundus images (different classes collide),
# so candidates are confirmed by Pearson correlation of 64x64 grayscale thumbnails
LABEL_RE = re.compile(r"^(\d+)_([A-Za-z])$")
issues, findings = [], []


def save(df, name):
    df.to_csv(T / f"audit_{name}.csv", index=False)
    return df


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string(index=False) if len(df) else "(none)")


# ---------------------------------------------------------------- 1. folder tree
print(f"Dataset root: {ROOT}\n")


def tree(path, depth, prefix=""):
    if depth > 3:
        return
    kids = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    dirs = [k for k in kids if k.is_dir()]
    files = [k for k in kids if k.is_file()]
    for d in dirs:
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"{prefix}{d.name}/  ({n} files)")
        tree(d, depth + 1, prefix + "    ")
    if files and depth > 1:
        print(f"{prefix}[{len(files)} files directly here, e.g. {files[0].name}]")
    elif files:
        for f in files:
            print(f"{prefix}{f.name}")


tree(ROOT, 1)

# ---------------------------------------------------------------- 2. file inventory
all_files = [p for p in ROOT.rglob("*") if p.is_file()]


def classify(p):
    parts = [s.lower() for s in p.relative_to(ROOT).parts[:-1]]
    joined = "/".join(parts)
    split = next((s for s in ("train", "test", "val") if s in parts), "none")
    if "ground" in joined or "mask" in joined or "label" in joined:
        kind = "mask"
    elif "original" in joined or "image" in joined:
        kind = "original"
    else:
        kind = "other"
    return split, kind


inv = pd.DataFrame(
    [{"path": p, "folder": str(p.parent.relative_to(ROOT)), "ext": p.suffix.lower() or "(none)",
      "split": classify(p)[0], "kind": classify(p)[1]} for p in all_files]
)
counts = save(inv.groupby(["folder", "split", "kind", "ext"]).size().rename("n_files").reset_index(), "file_counts")
show("Files per folder / split / kind / extension", counts)
non_img = inv[~inv["ext"].isin(IMG_EXTS)]
if len(non_img):
    issues.append(f"{len(non_img)} non-image file(s) present: {sorted(non_img['path'].map(lambda p: p.name).unique())}")

img_inv = inv[inv["ext"].isin(IMG_EXTS)].copy()
img_inv["stem"] = img_inv["path"].map(lambda p: p.stem)
img_inv["name"] = img_inv["path"].map(lambda p: p.name)
orig = img_inv[img_inv["kind"] == "original"].copy()
mask = img_inv[img_inv["kind"] == "mask"].copy()
unknown = img_inv[img_inv["kind"] == "other"]
if len(unknown):
    issues.append(f"{len(unknown)} image file(s) in folders not recognised as original/mask")
findings.append(
    f"Layout: originals in `{'`, `'.join(sorted(orig['folder'].unique()))}`; masks (vessel segmentation, "
    f"not class labels) in `{'`, `'.join(sorted(mask['folder'].unique()))}`; splits found: "
    f"{sorted(orig['split'].unique())}."
)

# ---------------------------------------------------------------- 3. label parsing
rows = []
for r in orig.itertuples():
    m = LABEL_RE.match(r.stem)
    letter = m.group(2).upper() if m else None
    rows.append({"split": r.split, "file": r.name, "id": m.group(1) if m else None, "letter": letter,
                 "label": config.LABEL_LETTER_TO_NAME.get(letter)})
labels = pd.DataFrame(rows)
bad = save(labels[labels["label"].isna()], "label_parse_failures")
show("Files whose label could not be parsed (letter not in A/D/G/N or bad name)", bad)
if len(bad):
    issues.append(f"{len(bad)} original image(s) with unparseable class label")
else:
    findings.append(f"Label parsing: all {len(labels)} original images match `<id>_<A|D|G|N>`.")
dup_ids = labels[labels.duplicated("id", keep=False)].sort_values("id")
if len(dup_ids):
    save(dup_ids, "duplicate_ids")
    findings.append(f"Numeric ids restart in each split ({dup_ids['id'].nunique()} ids appear in both train and test, "
                    "e.g. train/1_A and test/1_A are different images). Files must be keyed by split+filename, never id alone.")

# ---------------------------------------------------------------- 4. class counts
cc = labels.pivot_table(index="label", columns="split", values="file", aggfunc="count", fill_value=0)
cc["total"] = cc.sum(axis=1)
cc = save(cc.reset_index(), "class_counts")
show("Class counts per split", cc)
bal = cc[[c for c in cc.columns if c not in ("label", "total")]]
findings.append("Class counts per split:\n\n" + cc.to_markdown(index=False)
                + ("\n\nClasses are perfectly balanced within each split." if (bal.nunique() == 1).all()
                   else "\n\nClasses are NOT balanced."))


# ---------------------------------------------------------------- 5. per-file analysis
def analyse(path_str):
    path = Path(path_str)
    out = {"path": path_str, "size_bytes": path.stat().st_size, "error": None}
    try:
        out["md5"] = hashlib.md5(path.read_bytes()).hexdigest()
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            out["mode"], out["format"] = im.mode, im.format
            rgb = im.convert("RGB")
            out["width"], out["height"] = rgb.size
            out["phash"] = str(imagehash.phash(rgb))
        arr = np.asarray(rgb)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        out["thumb"] = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).ravel()
        out["brightness"] = float(gray.mean())
        out["contrast"] = float(gray.std())
        out["blur_lapvar"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        out["saturation"] = float(cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[..., 1].mean())
    except Exception as e:  # corrupt / unreadable
        out["error"] = f"{type(e).__name__}: {e}"
    return out


print("\nAnalysing images (hash, integrity, stats)...")
paths = [str(p) for p in pd.concat([orig, mask])["path"]]
res = pd.DataFrame(Parallel(n_jobs=-1)(delayed(analyse)(p) for p in paths))
res["path"] = res["path"].map(Path)
info = pd.concat([orig, mask]).merge(res, on="path")
info = info.merge(labels.rename(columns={"file": "name"})[["split", "name", "id", "label"]],
                  on=["split", "name"], how="left")

corrupt = save(info[info["error"].notna()][["split", "kind", "name", "error"]], "corrupt_files")
show("Corrupt / unreadable files", corrupt)
if len(corrupt):
    issues.append(f"{len(corrupt)} corrupt/unreadable file(s)")
else:
    findings.append("Integrity: every original and mask opened and decoded without error.")
ok = info[info["error"].isna()].copy()
o = ok[ok["kind"] == "original"].copy()
m = ok[ok["kind"] == "mask"].copy()

# ---------------------------------------------------------------- 6. dimensions / mode / size
dims = (o.assign(size=o["width"].astype(str) + "x" + o["height"].astype(str))
        .groupby(["split", "size", "mode", "format"]).size().rename("n").reset_index())
save(dims, "dimensions_mode")
show("Image dimensions / mode / format (originals)", dims)
fs = save(o.groupby("split")["size_bytes"].describe().reset_index(), "file_size_stats")
show("File size (bytes) distribution (originals)", fs)
if dims.groupby("split")["size"].nunique().max() > 1 or o.assign(s=o.width.astype(str) + o.height.astype(str))["s"].nunique() > 1:
    issues.append("Original images do not all share one resolution (resize needed)")
else:
    findings.append(f"All originals are {o.width.iloc[0]}x{o.height.iloc[0]} {o['mode'].iloc[0]}/{o['format'].iloc[0]}.")
mm = m.groupby(["split", "mode"]).size().rename("n").reset_index()
show("Mask modes", mm)

# ---------------------------------------------------------------- 7. image <-> mask matching
o_keys = set(zip(o["split"], o["stem"]))
m_keys = set(zip(m["split"], m["stem"]))
un = [{"side": "original_without_mask", "split": s, "id": i} for s, i in sorted(o_keys - m_keys)]
un += [{"side": "mask_without_original", "split": s, "id": i} for s, i in sorted(m_keys - o_keys)]
un = save(pd.DataFrame(un, columns=["side", "split", "id"]), "unmatched_masks")
show("Images without a mask (and vice versa)", un)
if len(un):
    issues.append(f"{len(un)} image/mask pairing mismatches")
else:
    findings.append(f"Masks: all {len(o_keys)} originals have a same-name mask in the same split.")
pair = o.merge(m, on=["split", "stem"], suffixes=("", "_m"))
dim_mis = pair[(pair.width != pair.width_m) | (pair.height != pair.height_m)]
if len(dim_mis):
    issues.append(f"{len(dim_mis)} image/mask pairs differ in size")

# ---------------------------------------------------------------- 8. duplicates
dup_rows = []
for md5, g in o.groupby("md5"):
    if len(g) > 1:
        dup_rows.append({"md5": md5, "files": " | ".join(f"{s}/{n}" for s, n in zip(g.split, g.name)),
                         "splits": ",".join(sorted(g.split.unique())), "labels": ",".join(sorted(g.label.astype(str).unique()))})
exact = save(pd.DataFrame(dup_rows, columns=["md5", "files", "splits", "labels"]), "exact_duplicates")
show("Exact duplicates (md5, originals)", exact)

o = o.reset_index(drop=True)
bits = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in o["phash"]])
ham = (bits[:, None, :] != bits[None, :, :]).sum(-1)
th = np.stack(o["thumb"].values).astype(np.float64)
th = (th - th.mean(1, keepdims=True)) / th.std(1, keepdims=True)
corr = th @ th.T / th.shape[1]
ii, jj = np.where(np.triu(ham <= PHASH_NEAR, k=1))
cand = pd.DataFrame({
    "file_a": o.split.values[ii] + "/" + o.name.values[ii], "file_b": o.split.values[jj] + "/" + o.name.values[jj],
    "label_a": o.label.values[ii], "label_b": o.label.values[jj], "phash_dist": ham[ii, jj],
    "thumb_corr": corr[ii, jj].round(4), "cross_split": o.split.values[ii] != o.split.values[jj],
    "same_md5": o.md5.values[ii] == o.md5.values[jj]})
save(cand.sort_values("thumb_corr", ascending=False).reset_index(drop=True), "near_duplicate_candidates")
near = cand[cand.thumb_corr >= CORR_CONFIRM].sort_values("thumb_corr", ascending=False).reset_index(drop=True)
near = save(near, "near_duplicates")
show(f"Near-duplicates (phash <= {PHASH_NEAR} AND thumbnail corr >= {CORR_CONFIRM})", near)
print(f"phash candidates: {len(cand)}, confirmed by pixel correlation: {len(near)}")
print("Candidate corr quantiles:", cand.thumb_corr.quantile([0, .5, .9, .99, 1]).round(3).to_dict())
leak = near[near["cross_split"]]
conflict = near[near["label_a"] != near["label_b"]]
extra = near[~near["same_md5"]]
if len(exact):
    issues.append(f"{len(exact)} exact-duplicate group(s) by md5 (labels: {sorted(set(','.join(exact['labels']).split(',')))}; "
                  f"{(exact['splits'].str.contains(',')).sum()} span splits): de-duplicate before CV or folds leak")
if len(extra):
    issues.append(f"{len(extra)} further near-duplicate pair(s) that are not byte-identical")
if len(leak):
    issues.append(f"{len(leak)} confirmed near-duplicate pair(s) across train/test (leakage)")
if len(conflict):
    issues.append(f"{len(conflict)} confirmed near-duplicate pair(s) with conflicting labels")
findings.append(f"Duplicates: {len(exact)} exact md5 groups; phash gave {len(cand)} candidates "
                f"(unreliable alone: it collides across classes), of which {len(near)} are confirmed by pixel "
                f"correlation >= {CORR_CONFIRM}; {len(leak)} confirmed pairs cross train/test.")

# ---------------------------------------------------------------- 9. quality stats
qcols = ["brightness", "contrast", "blur_lapvar", "saturation"]
q = o.groupby(["split", "label"])[qcols].agg(["mean", "std", "min", "max"])
q.columns = ["_".join(c) for c in q.columns]
q = save(q.reset_index(), "quality_by_class")
show("Brightness / contrast / blur / saturation by split & class", q)
overall = save(o[qcols].describe().T.reset_index().rename(columns={"index": "metric"}), "quality_overall")
show("Quality overall", overall)
lo = o["blur_lapvar"].quantile(0.01)
blurry = o[o.blur_lapvar <= lo][["split", "name", "label", "blur_lapvar"]]
save(blurry, "blurriest_1pct")
byc = o.groupby("label")[qcols].mean().round(2)
findings.append("Quality (whole image, black background included), mean per class:\n\n"
                + byc.reset_index().to_markdown(index=False)
                + "\n\nIf these differ strongly by class, acquisition-style differences could leak into "
                  "colour/intensity features (a shortcut to flag in EDA).")
spread = (byc.max() / byc.min()).round(2)
findings.append("Max/min ratio of class means: " + ", ".join(f"{k}={v}" for k, v in spread.items()))

# ---------------------------------------------------------------- 10. summary
save(info.drop(columns=["path"]).assign(path=info["path"].map(lambda p: str(p.relative_to(ROOT)))), "per_file")
md = ["# Dataset audit summary", "", f"Root: `{ROOT}`  ",
      f"Originals: {len(orig)}, masks: {len(mask)}, other files: {len(inv) - len(img_inv)}", "",
      "## Findings", ""] + [f"- {f}" if "\n\n" not in f else f"- {f.splitlines()[0]}\n\n" + "\n".join(f.splitlines()[1:])
                            for f in findings]
md += ["", "## Issues", ""] + ([f"- {i}" for i in issues] if issues else ["- None found."])
(RESULTS / "audit_summary.md").write_text("\n".join(md) + "\n")
print("\n" + "\n".join(md))
