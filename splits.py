import numpy as np
import pandas as pd
import os, time
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SPLIT_DIR = os.path.join(ROOT, "splits")

def make_splits(pool="main", fractions=(0.70, 0.15, 0.15 ), seed=0, overwrite=False):
    if os.path.exists(os.path.join(SPLIT_DIR, "test_ids.txt")) and not overwrite:
        raise SystemExit("splits already exist, change me to overwrite")

    plan = pd.read_csv(os.path.join(DATA_DIR, pool, "plan.csv"))
    ids = plan["episode_id"].to_numpy()

    #notes how episode ended
    summary = (pd.read_csv(os.path.join(DATA_DIR, pool, "summary.csv"), usecols=["episode_id", "stop_reason", "nav_done_time", "sim_duration", "best_ee_error"])
        .drop_duplicates("episode_id", keep="last").set_index("episode_id"))


    #check same episode available in arm and base dataset [over episode split]
    #contain makes unique if repeated and missing does set diff between the two
    for model in ("arm", "base"):
        contain = np.unique(np.load(os.path.join(DATA_DIR, pool, f"dataset_{model}.npz"))["episode_id"])
        missing = np.setdiff1d(ids, contain)
        if len(missing):
            print(f"dataset_{model}: no rows for {len(missing)} episode(s):")
            print(summary.loc[missing].to_string())
            assert (summary.loc[missing, "stop_reason"] != "reached").all(), "episode has 0 rows, check dataset"
    
    arm_ids = np.load(os.path.join(DATA_DIR, pool, "dataset_arm.npz"))["episode_id"]
    res_ids = np.load(os.path.join(DATA_DIR, pool, "dataset_arm_residual.npz"))["episode_id"]
    assert np.array_equal(arm_ids, res_ids), "dataset_arm and dataset_arm_residual rows differ - rebuild both"

    #shuffle id
    ids = np.random.default_rng(seed).permutation(ids)

    #splits (change in input of func)
    train_num = round(fractions[0] * len(ids)) 
    val_num = round(fractions[1] * len(ids))

    #split episode ids (rather than rows)
    splits = {"train": ids[:train_num], "val": ids[train_num:train_num+val_num], "test": ids[train_num+val_num:]}
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for name, s in splits.items():
        np.savetxt(os.path.join(SPLIT_DIR, f"{name}_ids.txt"), np.sort(s), fmt="%d")

    #adds new column split to df i.e. ep 103, val or 104, test
    plan["split"] = plan["episode_id"].map({i: name 
        for name, s in splits.items() for i in s})
    print(plan.groupby("split")[["d", "height"]].agg(["count", "mean", "std"]).round(2).to_string())


if __name__ == "__main__":
    make_splits()