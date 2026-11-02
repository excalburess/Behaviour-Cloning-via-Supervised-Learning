"""Torch MLP policy on top of Noah's pipeline: training side. Noah's files are not edited.

Pick a STEP at the bottom, then: pixi run python mlp.py
  "e_datasets"  : builds dataset_arm_e.npz (arm inputs + e = q_des - q -> torques) and dataset_qddes.npz (arm inputs -> qd_des)
  "dagger_pool" : turns the step logs of a closed-loop run (mlp_eval.py, SAVE_STEPS = True) into a data pool (DAgger)
  "train"       : trains MLPs on main train + EXTRA pools (e.g. the DART pools), saved as models/<NAME>_<dataset>.joblib
  "sweep"       : one-at-a-time hyperparameter sweep around the baseline
  "size"        : width/depth sweep, to find the smallest network that keeps the accuracy
Offline results go to Noah's outputs/results_offline.csv, in the same format as his models.
"""
import os
import glob
import time
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
import training
from training import (DATA_DIR, OUT_DIR, MODEL_DIR, TORQUE_LIMIT, load_dataset, load_splits, scale_labels,
                      predict_physical, metrics, time_single_row, append_results)
from pipeline import make_inputs, make_labels, build_dataset


class TorchMLP:
    #wraps a torch network so it looks like an sklearn model: .predict(raw numpy X) -> normalised numpy y
    def __init__(self, net, x_mean, x_std):
        self.net, self.x_mean, self.x_std = net.eval(), x_mean, x_std     #eval() switches dropout off

    def predict(self, X):
        X = torch.as_tensor((X - self.x_mean) / self.x_std, dtype=torch.float32)
        with torch.no_grad():
            return self.net(X).numpy().astype(np.float64)


training.TorchMLP = TorchMLP     #models saved before this file existed point to training.TorchMLP: lets them load


# ---------------------------------------------------------------- the MLP
MLP_HIDDEN = 256
MLP_LAYERS = 2
MLP_DROPOUT = 0.2
MLP_LR = 1e-3
MLP_EPOCHS = 100
MLP_PATIENCE = 10       #stop after this many epochs with no val improvement
MLP_BATCH = 1024


def fit_torch_mlp(train, val, y_mean, y_std, hidden=MLP_HIDDEN, layers=MLP_LAYERS, dropout=MLP_DROPOUT, lr=MLP_LR):
    torch.manual_seed(0)
    x_mean, x_std = train["X"].mean(axis=0), train["X"].std(axis=0)     #input scaling learnt on train only
    x_std[x_std < 1e-8] = 1.0
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    X_tr, X_va = to_t((train["X"] - x_mean) / x_std), to_t((val["X"] - x_mean) / x_std)
    y_tr, y_va = to_t((train["y"] - y_mean) / y_std), to_t((val["y"] - y_mean) / y_std)

    blocks, n = [], X_tr.shape[1]
    for _ in range(layers):                                             #hidden layers: Linear -> ReLU -> Dropout
        blocks += [nn.Linear(n, hidden), nn.ReLU(), nn.Dropout(dropout)]
        n = hidden
    net = nn.Sequential(*blocks, nn.Linear(hidden, y_tr.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
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
                                          "hidden": hidden, "layers": layers, "dropout": dropout, "lr": lr,
                                          "n_params": int(sum(p.numel() for p in net.parameters()))}


# ---------------------------------------------------------------- training run (Noah's training.run + extra pools)
def load_train(dataset, extra_pools=()):
    #train split of main, plus every episode of each extra pool (e.g. dart); val/test stay main only
    parts = [load_dataset("main", dataset, load_splits("train"))] + [load_dataset(pool, dataset) for pool in extra_pools]
    train = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "episode_id", "phase")}
    train["feature_names"], train["label_names"] = parts[0]["feature_names"], parts[0]["label_names"]
    return train


def run_mlp(dataset, model_name, extra_pools=(), **hp):
    #same steps and results row as training.run, but trains on main train + extra_pools and passes hp to fit_torch_mlp
    train = load_train(dataset, extra_pools)
    val = load_dataset("main", dataset, load_splits("val"))
    y_mean, y_std = scale_labels(train["y"])
    t0 = time.perf_counter()
    model, info = fit_torch_mlp(train, val, y_mean, y_std, **hp)
    fit_s = time.perf_counter() - t0

    bundle = {"model_name": model_name, "model": model, "y_mean": y_mean, "y_std": y_std, "dataset": dataset,
              "residual": dataset.endswith("residual"), "torque_limit": TORQUE_LIMIT,
              "feature_names": train["feature_names"], "label_names": train["label_names"]}
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, os.path.join(MODEL_DIR, f"{model_name}_{dataset}.joblib"))

    pred = predict_physical(bundle, val["X"])
    row = {"model": model_name, "dataset": dataset, **info, "fit_s": round(fit_s, 2),
           "infer_us": time_single_row(bundle, val["X"][0]), "train_rows": len(train["X"]), "val_rows": len(val["X"]),
           **metrics(val["y"], pred, y_std, train["label_names"], "val_")}
    if dataset.startswith("arm"):                              #also score on the full torque sent (as Noah does)
        raw_train = load_dataset("main", "arm", load_splits("train"))
        raw_val = load_dataset("main", "arm", load_splits("val"))
        tau_g = raw_val["y"] - val["y"] if bundle["residual"] else 0.0
        full = np.clip(pred + tau_g, -TORQUE_LIMIT, TORQUE_LIMIT)
        row.update(metrics(raw_val["y"], full, scale_labels(raw_train["y"])[1], raw_train["label_names"], "val_full_"))
    if dataset == "base":
        e = ((pred - val["y"]) / y_std) ** 2
        for phase in np.unique(val["phase"]):
            row[f"val_nmse_{phase}"] = float(e[val["phase"] == phase].mean())

    append_results(row)
    print(f"{model_name} {dataset:<10} val nmse {row['val_nmse']:.3f}  params {info['n_params']}  {row['infer_us']:.0f} µs/row")


# ---------------------------------------------------------------- extra datasets for the policy that tracks q_des (EPolicy)
def build_e_datasets(pool, stride=10):
    #same rows as dataset_arm.npz:  dataset_arm_e.npz = arm inputs + e0..e6 -> torques,  dataset_qddes.npz = arm inputs -> qd_des0..6
    folder = os.path.join(DATA_DIR, pool)
    reasons = pd.read_csv(os.path.join(folder, "summary.csv")).drop_duplicates(
        "episode_id", keep="last").set_index("episode_id")["stop_reason"]
    Xs, Es, ys, qds, ids, phases = [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(folder, "ep*.parquet"))):
        df = pd.read_parquet(f)
        eid = int(df["episode_id"].iloc[0])
        if reasons.get(eid) in ("unreachable", "step_cap"):          #same trimming as pipeline.build_dataset
            err = np.linalg.norm(df[["ee_x", "ee_y", "ee_z"]].to_numpy()
                                 - df[["target_x", "target_y", "target_z"]].to_numpy(), axis=1)
            df = df.iloc[: int(np.argmax(err <= err.min() + 0.002)) + 1]
        df = df[df["nav_done"].astype(bool)].iloc[::stride]          #arm rows only
        if len(df) == 0:
            continue
        Xs.append(make_inputs(df, "arm"))
        Es.append(pd.DataFrame({f"e{i}": df[f"q_des{i}"] - df[f"q{i}"] for i in range(7)}, index=df.index))
        ys.append(make_labels(df, "arm"))
        qds.append(pd.DataFrame({f"qd_des{i}": df[f"qd_des{i}"] for i in range(7)}, index=df.index))
        ids.append(np.full(len(df), eid))
        phases.append(df["phase"].astype(str).to_numpy())

    X, E = pd.concat(Xs, ignore_index=True), pd.concat(Es, ignore_index=True)
    y, qd = pd.concat(ys, ignore_index=True), pd.concat(qds, ignore_index=True)
    common = dict(episode_id=np.concatenate(ids), phase=np.concatenate(phases).astype(str))
    for name, inputs, labels in (("arm_e", pd.concat([X, E], axis=1), y), ("qddes", X, qd)):
        out = os.path.join(folder, f"dataset_{name}.npz")
        np.savez_compressed(out, X=inputs.to_numpy(np.float32), y=labels.to_numpy(np.float32), **common,
                            feature_names=np.array(inputs.columns, dtype=str), label_names=np.array(labels.columns, dtype=str))
        print(f"saved {out}: X {inputs.shape}, y {labels.shape}")


# ---------------------------------------------------------------- DAgger
def make_dagger_pool(run_name, pool, stride=10):
    #states from the model's own closed-loop run, labels = what the expert would have sent there (logged every step)
    src, dst = os.path.join(OUT_DIR, "closed_loop", run_name), os.path.join(DATA_DIR, pool)
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "ep*.parquet")))
    for f in files:
        df = pd.read_parquet(f)
        df["episode_id"] = int(os.path.basename(f)[2:7])          #the step logs don't store it, the file name has it
        df.to_parquet(os.path.join(dst, os.path.basename(f)), index=False)
    #stop_reason "dagger" so build_dataset keeps every step: the drift-away states are the ones DAgger needs labelled
    runs = pd.read_csv(os.path.join(OUT_DIR, f"closed_loop_{run_name}.csv"))
    runs = runs[runs["stop_reason"] != "crash"]
    pd.DataFrame({"episode_id": runs["episode_id"], "stop_reason": "dagger"}).to_csv(os.path.join(dst, "summary.csv"), index=False)
    print(f"{len(files)} episodes copied to {dst}")
    for model in ("arm", "base"):
        build_dataset(pool, model, stride, False)


# ---------------------------------------------------------------- settings
STEP = "train"
E_POOLS = ("main", "dart", "dart2")                  #e_datasets
DAGGER_RUN, DAGGER_POOL = "dagger_r1", "dagger_r1"   #dagger_pool: closed-loop NAME -> new data/<pool>/
NAME = "torch_mlp_dart2"                             #train: models/<NAME>_<dataset>.joblib
DATASETS = ("arm", "base", "arm_e", "qddes")         #train / size
EXTRA = ("dart", "dart2")                            #extra training pools; () = main only (torch_mlp), ("dart",) = torch_mlp_dart
HP = {}                                              #train: e.g. dict(hidden=128, dropout=0.1)
SWEEP_DATASET = "arm_e"
SWEEP = [dict(hidden=128), dict(hidden=512), dict(layers=1), dict(layers=3), dict(layers=4),
         dict(dropout=0.0), dict(dropout=0.1), dict(dropout=0.4), dict(lr=3e-4), dict(lr=3e-3)]
SIZES = [(16, 1), (32, 1), (64, 1), (128, 1), (16, 2), (32, 2), (64, 2), (128, 2)]    #(hidden, layers), dropout 0.1

if __name__ == "__main__":
    import mlp                                       #so saved models point to mlp.TorchMLP, not __main__.TorchMLP
    if STEP == "e_datasets":
        for pool in E_POOLS:
            mlp.build_e_datasets(pool)
    elif STEP == "dagger_pool":
        mlp.make_dagger_pool(DAGGER_RUN, DAGGER_POOL)
    elif STEP == "train":
        for dataset in DATASETS:
            mlp.run_mlp(dataset, NAME, EXTRA, **HP)
    elif STEP == "sweep":
        for change in SWEEP:
            hp = {"hidden": MLP_HIDDEN, "layers": MLP_LAYERS, "dropout": MLP_DROPOUT, "lr": MLP_LR, **change}
            mlp.run_mlp(SWEEP_DATASET, f"sweep2_h{hp['hidden']}_l{hp['layers']}_d{hp['dropout']}_lr{hp['lr']:g}", EXTRA, **hp)
    elif STEP == "size":
        for dataset in DATASETS:
            for hidden, layers in SIZES:
                mlp.run_mlp(dataset, f"size_h{hidden}_l{layers}", EXTRA, hidden=hidden, layers=layers, dropout=0.1)
