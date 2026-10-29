import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(ROOT, "splits")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "outputs")
RESULTS_CSV = os.path.join(OUT_DIR, "results_offline.csv")

ALPHAS = np.logspace(-2, 6, num=17) #ridge logvalues determine alpha which is a constant to reduce training error

def load_splits(name):
    return np.loadtxt(os.path.join(SPLIT_DIR, f"{name}_ids.txt"), dtype=int, ndmin=1)

def load_dataset(pool, dataset, ids=None):
    #dataset= arm, arm_res, base, ids = only rows whose id is in ids
    dataset = np.load(os.path.join(DATA_DIR, pool, f"dataset_{dataset}.npz"))
    #keep fills episodeID w true if no ids provided else checks if in dataset
    keep = np.full(len(dataset["episode_id"]), True) if ids is None else np.isin(dataset["episode_id"], ids)

    #returns kept data
    return {"X": dataset["X"][keep].astype(np.float64), "y": dataset["y"][keep].astype(np.float64),
            "episode_id": dataset["episode_id"][keep], "phase": dataset["phase"][keep],
            "feature_names": list(dataset["feature_names"]), "label_names": list(dataset["label_names"])}


#scales labels so every joint counts as equal in loss and metrics
def scale_labels(y_train):
    mean, std = y_train.mean(axis=0). y_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std

def dummy_regressor(train, y_mean, y_std):
    #baseline: mean/std of each output on train row (average)
    model = DummyRegressor().fit(train["X"], (train["y"] - y_mean) / y_std)
    return model, {"n_params": int(train["y"].shape[1])}

#alternative (more complex ridge fitting using pipeline and gridsearchCV)
# def fit_ridge(train, y_mean, y_std):
#     #finds best hyperparams(gridsearchcv) and scales inputs so all have same value(make_pipeline(standard_scaler())
#     pipeline = make_pipeline(StandardScaler(), Ridge()) 
#     search = GridSearchCV(pipeline, {"ridge__alpha": ALPHAS}, cv=GroupKFold(n_splits=5), scoring="neg_mean_squared_error")          #pipeline=estimator, 
#     search.fit(train["X"], (train["y"] - y_mean) / y_std, groups=train["episode_id"]) #groups is for kfold to see which episode each row belongs to
#     ridge = search.best_estimator_[-1]
#     return search.best_estimator_, {"alpha": float(search.best_params_["ridge__alpha"]),
#                                     "cv_nmse": float(-search.best_score_),                          #cross validated mean square error
#                                     "n_params": int(ridge.coef_.size + np.size(ridge.intercept_))}  #weight n_params

#try on alpha fit on train and keep best score (fit_ridge)
def fit_ridge(train, val, y_mean, y_std): #normalise vars
    y_tr = (train["y"] - y_mean) / y_std
    y_va = (val["y"] - y_mean) / y_std
    results = []                            #try each of 17 values 
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train["X"], y_tr)
        val_mse = float(np.mean((model.predict(val["X"]) - y_va) ** 2))
        results.append((val_mse, alpha, model))
    val_mse, alpha, model = min(results, key=lambda r: r[0]) #picks entry with smallest score compared by the first item in r with the blocks 'r'
    ridge = model[-1]
    return model, {"alpha": float(alpha), "select_nmse": val_mse,
                   "n_params": int(ridge.coef_.size + ridge.intercept_.size)}

def predict_physical(bundle, X):
    return bundle["model"].predict(X) * bundle["y_std"] + bundle["y_mean"]