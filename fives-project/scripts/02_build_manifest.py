"""02 - Build data/processed/manifest.csv (one row per kept image) and the cleaning log.

Reads the tables written by 01_audit_dataset.py. Nothing on disk under DATASET_PATH is touched.
Paths in the manifest are relative to config.DATASET_PATH so the manifest stays portable.
Decisions:
  * corrupt/unreadable image or mask, or missing mask -> dropped
  * exact duplicates (md5): if the group spans train/test, the train copies are dropped (test is
    never touched); otherwise the first copy (lowest id) is kept and the rest dropped
  * confirmed near-duplicate pairs that cross train/test (phash + pixel correlation, see 01) are
    also leakage, so the train member is dropped. Within-split near-duplicates that are not
    byte-identical are kept (not exact duplicates; reported only).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

T = config.TABLES_DIR
for f in ("audit_per_file.csv", "audit_near_duplicates.csv"):
    if not (T / f).exists():
        sys.exit(f"Missing {T / f}: run scripts/01_audit_dataset.py first.")

pf = pd.read_csv(T / "audit_per_file.csv", dtype={"id": "string"})
near = pd.read_csv(T / "audit_near_duplicates.csv")
orig = pf[pf.kind == "original"].copy()
mask = pf[pf.kind == "mask"].set_index(["split", "name"])
orig["key"] = orig["split"] + "/" + orig["name"]
orig["num_id"] = orig["stem"].str.extract(r"^(\d+)_")[0].astype(int)
orig = orig.sort_values(["split", "num_id"]).reset_index(drop=True)

log = []


def drop(keys, reason, detail=""):
    keys = set(keys) - {r["key"] for r in log}
    for k in sorted(keys):
        log.append({"key": k, "split": k.split("/")[0], "file": k.split("/")[1], "reason": reason, "detail": detail})


# 1. corrupt / missing mask
drop(orig[orig.error.notna()].key, "corrupt_image")
for r in orig.itertuples():
    m = mask.loc[(r.split, r.name)] if (r.split, r.name) in mask.index else None
    if m is None:
        drop([r.key], "missing_mask")
    elif isinstance(m["error"], str):
        drop([r.key], "corrupt_mask")

# 2. exact duplicates among what remains
alive = orig[~orig.key.isin({r["key"] for r in log})]
for md5, g in alive.groupby("md5"):
    if len(g) < 2:
        continue
    if g.split.nunique() > 1:  # crosses train/test: remove from train, never from test
        drop(g[g.split == "train"].key, "exact_duplicate_crosses_split", f"md5={md5}; train copy removed")
        rest = g[g.split == "test"]
    else:
        rest = g
    keep = rest.iloc[0].key  # sorted by (split, numeric id)
    drop(rest.key[rest.key != keep], "exact_duplicate", f"md5={md5}; kept {keep}")

# 3. confirmed near-duplicates that cross train/test -> drop the train member
for r in near[near.cross_split & ~near.same_md5].itertuples():
    tr = [k for k in (r.file_a, r.file_b) if k.startswith("train/")]
    other = [k for k in (r.file_a, r.file_b) if k not in tr]
    drop(tr, "near_duplicate_crosses_split", f"pair with {other[0]} (phash {r.phash_dist}, corr {r.thumb_corr}); train copy removed")

# 4. sanity: test never touched except genuine corruption
log_df = pd.DataFrame(log, columns=["key", "split", "file", "reason", "detail"]).drop(columns="key")
log_df.to_csv(T / "cleaning_log.csv", index=False)
test_drops = log_df[(log_df.split == "test") & ~log_df.reason.isin(["corrupt_image", "corrupt_mask", "missing_mask", "exact_duplicate"])]
assert test_drops.empty, test_drops

# manifest
kept = orig[~orig.key.isin({r["key"] for r in log})].copy()
D = config.DATASET_PATH
kept["image_path"] = kept.apply(lambda r: str(Path(r.split) / config.IMAGE_DIR_NAME / r["name"]), axis=1)
kept["mask_path"] = kept.apply(lambda r: str(Path(r.split) / config.MASK_DIR_NAME / r["name"]), axis=1)
assert all((D / p).exists() for p in kept.image_path) and all((D / p).exists() for p in kept.mask_path)
kept["image_id"] = kept["split"] + "_" + kept["stem"]  # ids restart per split, so prefix the split
kept["label_letter"] = kept["stem"].str[-1].str.upper()
kept["label_name"] = kept["label_letter"].map(config.LABEL_LETTER_TO_NAME)
kept["label_id"] = kept["label_name"].map(config.CLASS_TO_IDX)
assert kept[["label_name", "label_id"]].notna().all().all()
manifest = kept.rename(columns={"split": "official_split", "blur_lapvar": "blur_score"})[
    ["image_path", "mask_path", "image_id", "label_letter", "label_name", "label_id", "official_split",
     "width", "height", "blur_score", "brightness", "contrast"]].reset_index(drop=True)
manifest.to_csv(config.DATA_DIR / "manifest.csv", index=False)

print(f"Removed {len(log_df)} image(s):")
print(log_df.to_string(index=False) if len(log_df) else "(none)")
print(f"\nManifest: {len(manifest)} rows (from {len(orig)})")
ct = pd.crosstab(manifest.label_name, manifest.official_split)
ct = ct.reindex(config.CLASS_NAMES)
print("\nClass counts per split:\n", ct.to_string())
for s in ct.columns:
    c = ct[s]
    print(f"{s}: {'balanced' if c.nunique() == 1 else f'NOT balanced (min {c.min()}, max {c.max()}, ratio {c.max() / c.min():.2f})'}")
assert manifest.image_id.is_unique and manifest.image_path.is_unique
