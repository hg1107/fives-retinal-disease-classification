"""Model zoo: every classifier is placed after the preprocessing steps of src/preprocess.py in ONE flat Pipeline,
so imputer/scaler/filters/selector are refitted on the training part of each CV fold (no leakage).
Flat step names keep tuning simple: select__k, clf__C, ...
"""
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.preprocess import build_pipeline

SEED = 42


def baseline_models():
    """Default hyperparameters (except the few needed to run: iterations, probabilities, seeds, device)."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=SEED),  # multinomial (lbfgs) for 4 classes
        "RandomForest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=SEED),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu",
                                 eval_metric="mlogloss", random_state=SEED, n_jobs=4),
        "GaussianNB": GaussianNB(),
    }


def make_pipeline(clf, **prep_kwargs):
    """build_pipeline(**prep_kwargs) steps + final classifier named 'clf'."""
    return Pipeline(build_pipeline(**prep_kwargs).steps + [("clf", clf)])
