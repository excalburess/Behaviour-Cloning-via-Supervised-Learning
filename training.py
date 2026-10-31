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
import torch
from torch import nn


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(ROOT, "splits")
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "outputs")
RESULTS_CSV = os.path.join(OUT_DIR, "results_offline.csv")

ALPHAS = np.logspace(-2, 6, num=17) #Ridge penalty strengths to try: higher = smaller weights, less overfitting
TORQUE_LIMIT = 30.0 #FROM original arm_task.py

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
    mean, std = y_train.mean(axis=0), y_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std

def dummy_regressor(train, val, y_mean, y_std):
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




def predict_physical(bundled_data, X):
    return bundled_data["model"].predict(X) * bundled_data["y_std"] + bundled_data["y_mean"]
    #undoes normalisation in one place

# Code for the MLP, a wrapper is needed to use pytorch with the main code structure
class TorchMLP:
    #wraps a torch network so it looks like an sklearn model: .predict(raw numpy X) -> normalised numpy y
    def __init__(self, net, x_mean, x_std):
        self.net, self.x_mean, self.x_std = net.eval(), x_mean, x_std     #eval() switches dropout off

    def predict(self, X):
        X = torch.as_tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
        with torch.no_grad():
            return self.net(X).numpy().astype(np.float64)


MLP_HIDDEN = 256
MLP_DROPOUT = 0.3
MLP_EPOCHS = 100
MLP_PATIENCE = 20       #stop after this many epochs with no val improvement
MLP_BATCH = 1024
MLP_LR = 1e-3

def fit_torch_mlp(train, val, y_mean, y_std):
    torch.manual_seed(0)
    x_mean, x_std = train["X"].mean(axis=0), train["X"].std(axis=0)     #input scaling learnt on train only
    x_std[x_std < 1e-8] = 1.0
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    X_tr, X_va = to_t((train["X"] - x_mean) / x_std), to_t((val["X"] - x_mean) / x_std)
    y_tr, y_va = to_t((train["y"] - y_mean) / y_std), to_t((val["y"] - y_mean) / y_std)

    n_in, n_out = X_tr.shape[1], y_tr.shape[1]                          #arm: 26 -> 7, base: 8 -> 2
    net = nn.Sequential(nn.Linear(n_in, MLP_HIDDEN), nn.ReLU(), nn.Dropout(MLP_DROPOUT),
                        nn.Linear(MLP_HIDDEN, MLP_HIDDEN), nn.ReLU(), nn.Dropout(MLP_DROPOUT),
                        nn.Linear(MLP_HIDDEN, n_out))
    opt = torch.optim.Adam(net.parameters(), lr=MLP_LR)
    loss_fn = nn.MSELoss()

    best_mse, best_state, best_epoch = np.inf, None, 0
    for epoch in range(1, MLP_EPOCHS + 1):
        net.train()
        for idx in torch.randperm(len(X_tr)).split(MLP_BATCH):         #shuffled mini-batches
            opt.zero_grad()
            loss = loss_fn(net(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(net(X_va), y_va))
        if val_mse < best_mse:                                           #keep the best epoch on the val split
            best_mse, best_epoch = val_mse, epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        elif epoch - best_epoch >= MLP_PATIENCE:
            break

    net.load_state_dict(best_state)
    return TorchMLP(net, x_mean, x_std), {"select_nmse": best_mse, "best_epoch": best_epoch,
                                          "n_params": int(sum(p.numel() for p in net.parameters()))}

def metrics(y_true, y_pred, y_std, label_names, prefix):
    err = y_true-y_pred
    out = {f"{prefix}nmse": float(np.mean(err ** 2 / y_std ** 2)),
           f"{prefix}r2": float(r2_score(y_true, y_pred))} 
    for i, name in enumerate(label_names):
        out[f"{prefix}rmse_{name}"] = float(np.sqrt(np.mean(err[:, i] ** 2)))
    return out

def time_single_row(bundle, x_row, n=300):
    x_row = x_row.reshape(1, -1)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        predict_physical(bundle, x_row)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e6)

def append_results(row):
    os.makedirs(OUT_DIR, exist_ok=True)
    old = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()
    table = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    table.drop_duplicates(["model", "dataset"], keep="last").to_csv(RESULTS_CSV, index=False)


def describe_arm_data(): #describes how much arm data is hovering, and torque saturatoin    
    d = load_dataset("main", "arm", load_splits("train"))
    f = d["feature_names"]
    cols = []
    for axis in ["x", "y", "z"]:
        name = f"tgt_minus_ee_{axis}"
        position = f.index(name)
        cols.append(position)
    ee = d["X"][:, cols]
    print(f"arm train rows: {len(d['X'])} | hovering within 1 cm: {np.mean(np.linalg.norm(ee, axis=1) < 0.01):.1%}"
          f" | torque labels at the ±{TORQUE_LIMIT:.0f} N·m limit: {np.mean(np.abs(d['y']) > TORQUE_LIMIT - 0.1):.1%}")



def run(dataset, model_name, train_func):   #calls dummy regressor or ridge
    train = load_dataset("main", dataset, load_splits("train"))
    val = load_dataset("main", dataset, load_splits("val"))
    y_mean, y_std = scale_labels(train["y"])
    t0 = time.perf_counter()
    model, info = train_func(train, val, y_mean, y_std)               #info = small diagnostic
    fit_s = time.perf_counter() - t0                                #fit duration

    trained_model = {"model_name": model_name, "model": model, "y_mean": y_mean, "y_std": y_std, "dataset": dataset,
              "residual": dataset.endswith("residual"), "torque_limit": TORQUE_LIMIT,
              "feature_names": train["feature_names"], "label_names": train["label_names"]}
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(trained_model, os.path.join(MODEL_DIR, f"{model_name}_{dataset}.joblib"))

    pred = predict_physical(trained_model, val["X"])                   #validation inputs and converts predictions to Nm

    #where do values come from summary
    row = {"model": model_name, "dataset": dataset, **info, "fit_s": round(fit_s, 2),             #**info unpacks info dict(alpha, n_params)
           "infer_us": time_single_row(trained_model, val["X"][0]), "train_rows": len(train["X"]), "val_rows": len(val["X"]),   #infer us = time to predict row (microseconds)
           **metrics(val["y"], pred, y_std, train["label_names"], "val_")}

    #loads raw torque labels for the same val episodes so both arms get scored on torque sent
    if dataset.startswith("arm"):
        raw_train = load_dataset("main", "arm", load_splits("train"))
        raw_val = load_dataset("main", "arm", load_splits("val"))
        assert np.array_equal(raw_val["episode_id"], val["episode_id"]), "arm / arm_residual rows differ"
        tau_g = raw_val["y"] - val["y"] if trained_model["residual"] else 0.0     # residual label = torque - tau_g
        full = np.clip(pred + tau_g, -TORQUE_LIMIT, TORQUE_LIMIT)
        #compare predicted with full torque
        row.update(metrics(raw_val["y"], full, scale_labels(raw_train["y"])[1], raw_train["label_names"], "val_full_"))

    #averages error in base phase
    if dataset == "base":
        e = ((pred - val["y"]) / y_std) ** 2
        for phase in np.unique(val["phase"]):
            row[f"val_nmse_{phase}"] = float(e[val["phase"] == phase].mean())

    append_results(row)
    print(f"{model_name:>6} {dataset:<13} val nmse {row['val_nmse']:.3f}  r2 {row['val_r2']:.3f}"
          + (f"  full-torque nmse {row['val_full_nmse']:.3f}" if "val_full_nmse" in row else "")
          + (f"  alpha {row['alpha']:.0e}" if "alpha" in row else "") + f"  {row['infer_us']:.0f} µs/row")


if __name__ == "__main__":
    import training                                            #so the saved class is training.TorchMLP, not __main__.TorchMLP
    describe_arm_data()                                        #prints share of arm rows where hand within 1cm and, sahre of torque label sitting at saturated limit
    for dataset in ("arm", "arm_residual", "base"):
        for model_name, train_func in (("dummy", dummy_regressor), ("ridge", fit_ridge),
                                       ("torch_mlp", training.fit_torch_mlp)):
            run(dataset, model_name, train_func)
    print(f"\nfull table: {RESULTS_CSV}  |  models: {MODEL_DIR}")