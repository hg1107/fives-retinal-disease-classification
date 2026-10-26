"""Streamlit app for the FIVES retinal-disease classifier (research prototype, coursework only).

Reuses src/predict.py (predict_image) and src/features.py (preprocess / normalise). Every number shown is read from
models/*.json and results/ (tables, predictions, figures); nothing is hardcoded. Works without the dataset.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# CPU-only inference: must be set before torch is imported (predict.py imports it lazily, only if the model uses ResNet).
_META_PATH = ROOT / "models" / "best_model_meta.json"
try:
    if json.loads(_META_PATH.read_text()).get("feature_set") in ("resnet", "both"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
except (OSError, ValueError):
    pass

import altair as alt  # noqa: E402
import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from src import features as F  # noqa: E402
from src import predict  # noqa: E402

MODELS, RESULTS = ROOT / "models", ROOT / "results"
TABLES, FIGURES, PREDS = RESULTS / "tables", RESULTS / "figures", RESULTS / "predictions"
UNCERTAIN_BELOW = 0.5
DISCLAIMER = "Research prototype for a university assignment. Not a medical device and not for diagnosis."

st.set_page_config(page_title="FIVES retinal classifier", page_icon="👁", layout="wide")


# ------------------------------------------------------------------ results readers
@st.cache_data
def read_json(path):
    return json.loads(Path(path).read_text())


@st.cache_data
def read_table(name):
    return pd.read_csv(TABLES / name)


@st.cache_resource
def load_model():
    """Load the saved model once per server process (predict.py keeps it in its own _STATE; not duplicated here)."""
    st_ = predict._load()
    return st_["meta"], [st_["classes"][i] for i in sorted(st_["classes"])]


def final_row(meta):
    """Row of tune_before_after.csv for the saved final model (test metrics and CV stats)."""
    t = read_table("tune_before_after.csv")
    return t[t["candidate"] == meta["candidate"]].iloc[0]


def normal_recall(meta):
    return float(final_row(meta)["test_recall_Normal"])


def robustness():
    """-> (clean macro-F1, {perturbation: macro-F1}) from abl_robustness.csv."""
    t = read_table("abl_robustness.csv").set_index("perturbation")["macro_f1"]
    return float(t["clean"]), {k: float(v) for k, v in t.items() if k != "clean"}


def pct(x):
    return f"{100 * x:.0f}%"


# ------------------------------------------------------------------ image handling
def decode_upload(data):
    """Bytes -> RGB uint8 array (same decoder as features.read_rgb). Raises ValueError if it is not a readable image."""
    try:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    except cv2.error:
        arr = None
    if arr is None:
        raise ValueError("The file could not be read as an image (it may be corrupt or not a PNG/JPG).")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def check_looks_like_fundus(img):
    """Cheap sanity check for blank / non-fundus input; features.preprocess alone accepts e.g. an all-white frame.

    A colour fundus photo has a visible retina region with real contrast and a red-dominant colour balance.
    """
    small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = small.mean(axis=2)
    fov = gray > F.FOV_THRESH
    if fov.mean() < 0.1:
        raise ValueError("The image is almost completely black; no retina was found.")
    if gray[fov].std() < 3:  # darkest real images in the dataset have std ~5
        raise ValueError("The image is blank or has almost no contrast, so it does not look like a fundus photograph.")
    r, b = small[..., 0][fov].mean(), small[..., 2][fov].mean()
    if r < 1.2 * b:
        raise ValueError("The colours do not look like a retinal fundus photograph (expected a red/orange retina).")


def to_display(img, size=512):
    k = size / max(img.shape[:2])
    return img if k >= 1 else cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


@st.cache_data(show_spinner=False, max_entries=16)
def analyse(data, use_norm):
    """Bytes -> dict(label, probs, original, pre, norm). Raises ValueError on unusable input."""
    img = decode_upload(data)
    check_looks_like_fundus(img)
    pre = F.preprocess(img)  # raises ValueError if no field of view is found
    label, probs = predict.predict_image(img)
    out = {"label": label, "probs": probs, "original": to_display(img), "pre": pre["rgb"], "norm": None}
    if use_norm:
        out["norm"] = F.normalise(pre)["rgb"]
    return out


def sample_images():
    """{class name: path} with one test image per class, or {} when the dataset is not available (e.g. deployed app)."""
    try:
        folder = config.DATASET_PATH / "test" / config.IMAGE_DIR_NAME
        if not folder.is_dir():
            return {}
        files = sorted(folder.glob(f"*{config.IMAGE_EXT}"))
        out = {}
        for letter, name in config.LABEL_LETTER_TO_NAME.items():
            match = [f for f in files if f.stem.endswith(f"_{letter}")]
            if match:
                out[name] = match[0]
        return out
    except OSError:
        return {}


# ------------------------------------------------------------------ pages
def page_predict():
    meta, classes = load_model()
    st.title("Predict")
    st.warning(DISCLAIMER)

    up = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])
    samples = sample_images()
    choice = None
    if samples:
        choice = st.selectbox("...or use a sample image", ["(none)"] + [f"{n} ({p.name})" for n, p in samples.items()])
        if choice == "(none)":
            choice = None

    data, source = None, None
    if up is not None:
        data, source = up.getvalue(), up.name
    elif choice:
        path = samples[choice.split(" (")[0]]
        try:
            data, source = path.read_bytes(), path.name
        except OSError as e:
            st.error(f"Could not read the sample image: {e}")
            return
    if data is None:
        st.info("Upload a fundus photograph (PNG/JPG) to get a prediction.")
        return

    try:
        with st.spinner("Analysing..."):
            res = analyse(data, meta["feature_set"] == "handcrafted_norm")
    except ValueError as e:
        st.error(f"Cannot classify this image: {e}")
        return
    except Exception as e:  # noqa: BLE001 - never show a traceback to the user
        st.error(f"Something went wrong while analysing this image ({type(e).__name__}). Please try a different file.")
        return

    label, probs = res["label"], res["probs"]
    top = max(probs.values())

    if label == "Normal":
        st.warning(
            f"Normal predictions are unreliable. On the test set only {pct(normal_recall(meta))} of Normal images were "
            "recognised, because Normal train and test images come from different sources."
        )
    if top < UNCERTAIN_BELOW:
        st.warning("The model is uncertain about this image.")
    with st.expander("Known limitations", expanded=False):
        clean, perturbed = robustness()
        noise = {k: v for k, v in perturbed.items() if k.startswith("noise")}
        blur = {k: v for k, v in perturbed.items() if k.startswith("blur")}
        st.write(
            "The model is not robust to blur or noise, and was not tested on heavy compression, so assume it is also "
            f"fragile there. Macro-F1 on the test set fell from {clean:.2f} to "
            + ", ".join(f"{v:.2f} ({k})" for k, v in noise.items())
            + " under added noise, and to "
            + ", ".join(f"{v:.2f} ({k})" for k, v in blur.items())
            + " under blur."
        )

    st.subheader("Images")
    cols = st.columns(3 if res["norm"] is not None else 2)
    cols[0].image(res["original"], caption=f"Input ({source})", width="stretch")
    cols[1].image(res["pre"], caption="Preprocessed (FOV crop and mask)", width="stretch")
    if res["norm"] is not None:
        cols[2].image(res["norm"], caption="Normalised (what the colour/texture features see)", width="stretch")

    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Predicted class", label)
    c2.metric("Top confidence", f"{top:.1%}")
    df = pd.DataFrame({"class": classes, "probability": [probs[c] for c in classes]})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("class:N", sort=classes, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["class", alt.Tooltip("probability:Q", format=".3f")],
        )
    )
    st.altair_chart(chart, width="stretch")


def confusion(meta):
    p = pd.read_csv(PREDS / f"tune__{meta['candidate']}.csv")
    names = {int(k): v for k, v in read_json(MODELS / "classes.json").items()}
    order = [names[i] for i in sorted(names)]
    return pd.crosstab(p["y_true"].map(names), p["y_pred"].map(names)).reindex(index=order, columns=order, fill_value=0)


def page_performance():
    meta, _ = load_model()
    row = final_row(meta)
    hp = read_json(RESULTS / "best_hyperparameters.json")
    st.title("Model performance")
    st.warning(DISCLAIMER)

    st.subheader("Final model")
    st.write(
        f"**{meta['candidate']}** ensemble on the **{meta['feature_set']}** feature set "
        f"({len(meta['feature_columns'])} features). Selected by cross-validation only; the test split was used once for reporting. "
        f"Ensemble members were tuned with {hp['n_iter']}-iteration randomized search on `{hp['experiment']}` features."
    )
    c = st.columns(5)
    c[0].metric("CV macro-F1", f"{meta['cv_f1_macro']:.3f}", help=f"{meta['cv_estimate']}, ± {row['cv_f1_std']:.3f} std")
    c[1].metric("Test macro-F1", f"{row['test_f1_macro']:.3f}")
    c[2].metric("Test accuracy", f"{row['test_accuracy']:.3f}")
    c[3].metric("Test macro-recall", f"{row['test_recall_macro']:.3f}")
    c[4].metric("Test ROC-AUC (OvR)", f"{row['test_roc_auc_ovr']:.3f}")
    st.caption(f"CV: {meta['cv_estimate']} on the training split, mean ± std = {row['cv_f1_mean']:.3f} ± {row['cv_f1_std']:.3f}.")
    with st.expander("Tuned hyperparameters"):
        for name, m in hp["models"].items():
            st.markdown(f"**{name}** (inner-CV macro-F1 {m['best_inner_cv_f1']:.3f})")
            st.json(m["params"], expanded=False)

    st.subheader("Baseline vs tuned vs ensembles")
    t = read_table("tune_before_after.csv")
    show = pd.DataFrame(
        {
            "stage": t["stage"],
            "model": t["candidate"],
            "CV macro-F1": t["cv_f1_mean"].round(3),
            "CV std": t["cv_f1_std"].round(3),
            "CV accuracy": t["cv_accuracy_mean"].round(3),
            "test macro-F1": t["test_f1_macro"].round(3),
            "test accuracy": t["test_accuracy"].round(3),
            "test ROC-AUC": t["test_roc_auc_ovr"].round(3),
            "test Normal recall": t["test_recall_Normal"].round(2),
            "CV verdict vs baseline": t["cv_verdict"].fillna(""),
        }
    )
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Baseline CV is 5-fold; tuned and ensemble CV is nested 5-fold. Test numbers come from the single official test split.")
    with st.expander("All baseline models x feature sets"):
        cmp_ = read_table("cmp_comparison_table.csv")
        st.dataframe(
            cmp_[["experiment", "model", "cv_f1_macro_mean", "cv_f1_macro_std", "test_f1_macro", "test_accuracy", "test_roc_auc_ovr"]]
            .round(3)
            .rename(columns=lambda s: s.replace("_", " ")),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Test confusion matrix and per-class recall")
    cm = confusion(meta)
    recall = pd.Series(np.diag(cm.values) / cm.sum(axis=1).values, index=cm.index)
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.imshow(cm.values / cm.sum(axis=1).values[:, None], cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cm)), [s.replace(" ", "\n") for s in cm.columns])
        ax.set_yticks(range(len(cm)), [s.replace(" ", "\n") for s in cm.index])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(cm)):
            for j in range(len(cm)):
                v = cm.values[i, j]
                ax.text(j, i, v, ha="center", va="center", color="white" if v / cm.values[i].sum() > 0.5 else "black")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Counts; colour shows the row-normalised share (recall on the diagonal).")
    with right:
        rdf = pd.DataFrame({"class": list(recall.index), "recall": recall.values})
        bars = (
            alt.Chart(rdf)
            .mark_bar()
            .encode(
                x=alt.X("class:N", sort=list(cm.index), title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("recall:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=["class", alt.Tooltip("recall:Q", format=".2f")],
            )
        )
        st.altair_chart(bars, width="stretch")
        st.caption(f"Based on {int(cm.values.sum())} test images; one recall value has a standard error of roughly 0.07.")

    st.subheader("Honest reading of these numbers")
    three = float(read_table("abl_normal_excluded.csv").iloc[0]["macro_f1_3cls"])
    g = read_table("tune_gain_vs_noise.csv")
    within = g["cv_verdict"].isin(["gain within CV noise", "no CV gain"]).all()
    best = g.loc[g["cv_gain_f1"].idxmax()]
    st.markdown(
        f"- **CV vs test:** CV macro-F1 is {meta['cv_f1_macro']:.3f} but test macro-F1 is {row['test_f1_macro']:.3f}, "
        f"a gap of {meta['cv_f1_macro'] - row['test_f1_macro']:.3f}. CV cannot see the Normal source shift, so it is optimistic.\n"
        f"- **Normal is the problem class:** test recall for Normal is {row['test_recall_Normal']:.2f}, against "
        f"{row['test_recall_AMD']:.2f} (AMD), {row['test_recall_Diabetic Retinopathy']:.2f} (DR) and {row['test_recall_Glaucoma']:.2f} (Glaucoma).\n"
        f"- **Disease classes only:** test macro-F1 over AMD, DR and Glaucoma is {three:.3f}.\n"
        f"- **Tuning and ensembling:** the largest CV gain over a baseline is {best['cv_gain_f1']:+.3f} ({best['candidate']}), "
        f"which is {best['gain_over_ref_std']:.2f} of the baseline's CV standard deviation. "
        + ("All gains were within CV noise, so the ranking of the top candidates is not statistically meaningful."
           if within else "Not every gain was within CV noise; see the table above.")
    )
    st.info(DISCLAIMER)


def page_eda():
    st.title("Dataset and EDA")
    st.markdown(
        "**FIVES** is a fundus image dataset for AI-based vessel segmentation, hosted on **Figshare**. "
        "Each image carries a disease label (Normal, AMD, Diabetic Retinopathy, Glaucoma) in its file name, "
        "plus a vessel mask that this project uses only to sanity-check its own vessel features."
    )
    counts = read_table("audit_class_counts.csv")
    st.subheader("Class distribution (as downloaded)")
    long = counts.melt(id_vars="label", value_vars=["train", "test"], var_name="split", value_name="images")
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("images:Q"),
            color="split:N",
            xOffset="split:N",
            tooltip=["label", "split", "images"],
        )
    )
    st.altair_chart(chart, width="stretch")
    log_path = TABLES / "cleaning_log.csv"
    if log_path.exists():
        removed = pd.read_csv(log_path)
        st.caption(
            f"{int(counts[['train', 'test']].to_numpy().sum())} images downloaded; {len(removed)} removed before modelling "
            f"({', '.join(f'{n} {r}' for r, n in removed['reason'].value_counts().items())}). "
            "The class-balance figure below shows the cleaned set."
        )

    figs = sorted(FIGURES.glob("eda_*.png"))
    if figs:
        st.subheader("Figures")
        pick = st.selectbox("Figure", figs, format_func=lambda p: p.stem.replace("eda_", "", 1).replace("_", " "))
        st.image(str(pick), width="stretch")
    insights = RESULTS / "eda_insights.md"
    if insights.exists():
        with st.expander("EDA insights (generated)"):
            st.markdown(insights.read_text())

    st.subheader("Citation")
    st.markdown(
        "Jin K. et al., *FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation*, Sci Data **9**, 475 (2022). "
        "DOI: [10.6084/m9.figshare.19688169](https://doi.org/10.6084/m9.figshare.19688169). Hosted on Figshare."
    )


def page_about():
    meta, _ = load_model()
    st.title("About")
    st.warning(DISCLAIMER)
    st.subheader("Pipeline")
    st.markdown(
        "1. Resize, detect the retina's field of view (FOV), crop, apply a circular FOV mask and CLAHE on the green channel.\n"
        "2. For the `handcrafted_norm` feature set, correct illumination and standardise each colour channel per image.\n"
        f"3. Extract {len(meta['feature_columns'])} handcrafted features (colour, texture, vessel, bright-region).\n"
        f"4. Classify with the saved {meta['candidate']} ensemble (scikit-learn pipeline with all fitting inside CV folds).\n"
        "5. Output class probabilities for Normal, AMD, Diabetic Retinopathy and Glaucoma."
    )
    st.subheader("Limitations")
    auc = read_table("abl_train_vs_test_auc.csv")
    n_auc = auc[(auc["class"] == "Normal") & (auc["feature_set"] == meta["feature_set"])]["train_vs_test_auc"].mean()
    r = float(read_table("feat_vessel_sanity.csv").set_index("feature").loc["vessel_density", "pearson_r"])
    st.markdown(
        f"- **Normal train/test source shift.** Normal training and test images come from different sources: a classifier can "
        f"tell them apart with AUC about {n_auc:.2f} (much lower for the other classes). Normal test recall is only "
        f"{normal_recall(meta):.2f}, and CV cannot reveal this. Treat Normal predictions as unreliable.\n"
        "- **Corner-glyph shortcut.** A small overlay glyph in the image corner marks the image source and could be used as a label "
        "shortcut. It is removed by masking everything outside a circular FOV before feature extraction.\n"
        f"- **Vessel features are only a proxy.** Image-derived vessel density agrees only moderately with the ground-truth masks "
        f"(Pearson r = {r:.2f}).\n"
        "- **Small data and one split.** About 600 training and 200 test images, one train/test partition.\n"
        "- **Not robust** to blur or noise (see Known limitations on the Predict page)."
    )
    st.info(DISCLAIMER)


PAGES = {"Predict": page_predict, "Model performance": page_performance, "Dataset and EDA": page_eda, "About": page_about}

page = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption(DISCLAIMER)
PAGES[page]()
