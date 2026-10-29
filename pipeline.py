import os, time, glob, argparse, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc
import multiprocessing as mp
from tqdm import tqdm


#DIRECTORIES
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")

#RANGES
TRAIN = dict(dist=(0.8, 2.5), angle=(-np.pi, np.pi), height=(0.15, 1.10),
             start_xy=(-1.5, 1.5), start_yaw=(-np.pi, np.pi),
             arm_offset=0.25, arm_offset_prob=0.7)

LIMITED_JOINTS = (1, 3, 5)

#Out Of Range Sets
OOR = {
    "train":     TRAIN,
    "oor_far":   {**TRAIN, "dist": (2.5, 3.5)},
    "oor_high":  {**TRAIN, "height": (1.1, 1.3)},
    "oor_start": {**TRAIN, "start_xy": (1.5, 3.0)},
    "oor_arm":   {**TRAIN, "arm_offset": 0.5},
}

N_DIMS = 14 #distance, angle, height, start_x, start_y, start_yaw, offset_on, offset0....6



def make_plan(name, n, first_id, seed, ranges, method, noise_std=0.0):
    gen = np.random.default_rng(seed)
    if method == "lhs":
        u = qmc.LatinHypercube(d=14, rng=gen).random(n)
    else:
        u = gen.random((n, 14))

    #limits
    d_lo, d_hi = ranges["dist"]
    a_lo, a_hi = ranges["angle"]
    h_lo, h_hi = ranges["height"]
    s_lo, s_hi = ranges["start_xy"]
    y_lo, y_hi = ranges["start_yaw"]

    #values

    d = np.sqrt(d_lo ** 2 + u[:, 0] * (d_hi ** 2 - d_lo ** 2)) #distance √( d_lo² + u × (d_hi² − d_lo²) )
    angle = a_lo + u[:, 1] * (a_hi - a_lo)
    height = h_lo + u[:, 2] * (h_hi - h_lo)
    start_x = s_lo + u[:, 3] * (s_hi - s_lo)
    start_y = s_lo + u[:, 4] * (s_hi - s_lo)
    start_yaw = y_lo + u[:, 5] * (y_hi - y_lo)

    A = ranges["arm_offset"]
    offsets = -A + 2 * A * u[:, 7:14] # u [:, 7:14] takes columns 7 to 13 of the random table, which is 7 columns, one per arm joint randomises offset
    offsets[u[:, 6] >= ranges["arm_offset_prob"]] = 0.0

    ids = first_id + np.arange(n)
    plan = pd.DataFrame({
        "episode_id": ids, "pool": name, "method": method, "plan_seed": seed,
        "target_x": start_x + d * np.cos(start_yaw + angle),
        "target_y": start_y + d * np.sin(start_yaw + angle),
        "target_z": height,
        "start_x": start_x, "start_y": start_y, "start_yaw": start_yaw,
        "d": d, "angle": angle, "height": height,
        "noise_std": float(noise_std), "noise_seed": ids,
    })
    for i in range(7):
        plan[f"arm_offset{i}"] = offsets[:, i]
    return plan


def plot_coverage(df, out_png, title, colour_by=None):
    #Shows targets relative to robot, histograms show episode per distance for quality assurance
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    labels = [None] if colour_by is None else sorted(df[colour_by].dropna().unique())
    for label in labels:
        g = df if label is None else df[df[colour_by] == label]
        ax[0, 0].scatter(g["d"] * np.cos(g["angle"]), g["d"] * np.sin(g["angle"]), s=12, label=label)
        ax[0, 1].scatter(g["d"], g["height"], s=12, label=label)
    ax[0, 0].scatter([0], [0], marker="^", s=80, c="k", label="start (facing +x)")
    ax[0, 0].set(xlabel="forward of start (m)", ylabel="left of start (m)", title="Top view (robot frame)")
    ax[0, 0].set_aspect("equal")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].set(xlabel="horizontal distance d (m)", ylabel="target height (m)", title="Side view")
    ax[1, 0].hist(df["d"], bins=20)
    ax[1, 0].set(xlabel="horizontal distance d (m)", ylabel="episodes")
    ax[1, 1].hist(df["height"], bins=20)
    ax[1, 1].set(xlabel="target height (m)", ylabel="episodes")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("saved", out_png)

def run_one_row(row): #unpacks row to fill in each of run_episodes arguments, adds info to summary
    os.environ["ROBOENV_HEADLESS"] = "1"
    t0 = time.perf_counter()
    eid = int(row["episode_id"])
    summary: dict = {}          

    try:
        from robotnik_arm_task import run_episode            # imported in the worker
        df, summary = run_episode(
            target=[row["target_x"], row["target_y"], row["target_z"]],
            start_xy=[row["start_x"], row["start_y"]], start_yaw=float(row["start_yaw"]),
            start_arm_offset=np.array([row[f"arm_offset{i}"] for i in range(7)], dtype=float),
            seed=eid, noise_seed=int(row["noise_seed"]), noise_std=float(row["noise_std"]),
            headless=True, verbose=False)
        df["episode_id"] = eid
        path = os.path.join(DATA_DIR, row["pool"], f"ep{eid:05d}.parquet")
        df.to_parquet(path + ".tmp", index=False)            
        os.replace(path + ".tmp", path)
    except Exception as e:
        summary = {"stop_reason": "crash", "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1500:].replace("\n", " | ")}
        
    summary.update(episode_id=eid, pool=row["pool"], d=row["d"], angle=row["angle"],
        height=row["height"], wall_time_s=time.perf_counter() - t0)
    return summary


def collect_data(name, workers=1, limit=None):
    folder = os.path.join(DATA_DIR, name)  
    plan = pd.read_csv(os.path.join(folder, "plan.csv"))
    done = np.array([os.path.exists(os.path.join(folder, f"ep{i:05d}.parquet")) for i in plan["episode_id"]]) #checks per episode id if file exists {i:05d} pads to 5 digits so episode 7: 00007
    todo = plan[~done].head(limit) if limit else plan[~done]            #every episode set to not done
    rows = todo.to_dict("records")                                      #turns table into list of dictionary
    print(f"[{name}] {done.sum()} done, running {len(rows)} of {len(plan)} planned, workers={workers}", flush=True) #flush prints at this line no matter waht
    if not rows: #if collected data dont do anything
        return
    
 
    summary_csv = os.path.join(folder, "summary.csv")                                               #results go here
    table = pd.read_csv(summary_csv).to_dict("records") if os.path.exists(summary_csv) else []      #adds data from prev run if exited
    pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=1) if workers > 1 else None       #starts a new process
    results = pool.imap_unordered(run_one_row, rows) if pool else map(run_one_row, rows)            #hands episodes to workers and gives summary as soon as any worker finishes
    t0 = time.perf_counter()                                                                        #time for speed and eta                                         
    try:
        bar = tqdm(results, total=len(rows), desc=name, unit="ep")                                  #progress bar using tdqm
        for s in bar:
            table.append(s)
            pd.DataFrame(table).to_csv(summary_csv + ".tmp", index=False)
            os.replace(summary_csv + ".tmp", summary_csv)
            bar.set_postfix(ep=s["episode_id"], last=s["stop_reason"])
            if s["stop_reason"] == "crash":
                tqdm.write(f"[{name}] episode {s['episode_id']} crashed: {s.get('error')}")
    finally:
        if pool:
            pool.terminate()

def qa(name):
    folder = os.path.join(DATA_DIR, name)
    plan = pd.read_csv(os.path.join(folder, "plan.csv"))
    summ = pd.read_csv(os.path.join(folder, "summary.csv")).drop_duplicates("episode_id", keep="last")
    files = sorted(glob.glob(os.path.join(folder, "ep*.parquet")))
 
    print(f"\n===== QA {name}: planned {len(plan)}, finished {len(summ)}, files {len(files)} =====")                        #amount of files in collection completed
    print(summ["stop_reason"].value_counts(normalize=True).round(2).to_string())                                               #normalised stop reason i.e. reachable = 0.8, unreached = 0.2.....
    print("\nby height:\n", pd.crosstab(pd.cut(summ["height"], 5), summ["stop_reason"], normalize="index").round(2))        #height summary stop reason
    print("\nby distance:\n", pd.crosstab(pd.cut(summ["d"], 5), summ["stop_reason"], normalize="index").round(2))           #distance summary stop reason (crosstab= frequency)
    size_mb = sum(os.path.getsize(f) for f in files) / 1e6                                                                  
    print(f"\n{size_mb / max(len(files), 1):.1f} MB/episode, {summ['wall_time_s'].mean():.0f} s/episode")                      #per ep size and time
 
    flags = [f"crash: episode {e}" for e in summ.loc[summ["stop_reason"] == "crash", "episode_id"]]   #where stop reason = crash get the episode id
    for f in files:
        df = pd.read_parquet(f) 
        numeric = df.loc[:, ~df.columns.str.startswith("lim_margin")].select_dtypes("number")    #drop where near limit margin
        if numeric.isna().any().any():
            flags.append(f"NaN in {os.path.basename(f)}")
        if not np.allclose(df["t"], df["step"] * 0.001):
            flags.append(f"t != step*0.001 in {os.path.basename(f)}")
        if (df[[f"finger{i}" for i in range(6)]] - 0.1).abs().max().max() > 0.01:  #checks for finger drift more than 0.01 rad
            flags.append(f"finger drift in {os.path.basename(f)}")
    success = (summ["stop_reason"] == "reached").mean()
    if success < 0.8:
        flags.append(f"expert success only {success:.0%}")
 
    plot_coverage(summ, os.path.join(OUT_DIR, f"outcomes_{name}.png"), f"{name}: outcomes", colour_by="stop_reason") #where target were coloured by result
    print("\n".join("RED FLAG: " + f for f in flags) if flags else f"QA PASSED (expert success {success:.0%})")

def make_features(df, model, residual=False):
    #transforms record into model input (x) and label (y)
    yaw = df["base_yaw"].to_numpy()
    c, s = np.cos(yaw), np.sin(yaw)
    tx, ty, tz = df["target_x"].to_numpy(), df["target_y"].to_numpy(), df["target_z"].to_numpy()
    dx, dy = tx - df["base_x"].to_numpy(), ty - df["base_y"].to_numpy()
    X = {"tgt_rel_x": c * dx + s * dy, "tgt_rel_y": -s * dx + c * dy, "tgt_z": tz}   #how far target is forward then left
 
    if model == "base":   #how far forward target is from base
        vx, vy = df["base_vx"].to_numpy(), df["base_vy"].to_numpy()
        X["tgt_dist"] = np.hypot(X["tgt_rel_x"], X["tgt_rel_y"])                #how far and how much to turn to face it
        X["tgt_bearing"] = np.arctan2(X["tgt_rel_y"], X["tgt_rel_x"])
        X["v_forward"], X["v_left"] = c * vx + s * vy, -s * vx + c * vy
        X["yaw_rate"] = df["base_wz"].to_numpy()
        y = {"cmd_right": df["cmd_wheel0"], "cmd_left": df["cmd_wheel1"]}      #wheel speeds expert chose    
    else:
        for i in range(7):                  
            X[f"q{i}"] = df[f"q{i}"].to_numpy() 
            X[f"qd{i}"] = df[f"qd{i}"].to_numpy()   
        for i in LIMITED_JOINTS:
            X[f"lim_margin{i}"] = df[f"lim_margin{i}"].to_numpy()  #dinstace to limit              
        for part in ("shoulder", "ee"):                                                     #shoulder to target
            px, py = tx - df[f"{part}_x"].to_numpy(), ty - df[f"{part}_y"].to_numpy()           #how far target is from shoulder
            X[f"tgt_minus_{part}_x"] = c * px + s * py                                                 #turn arrow into robot direction
            X[f"tgt_minus_{part}_y"] = -s * px + c * py
            X[f"tgt_minus_{part}_z"] = tz - df[f"{part}_z"].to_numpy()
        if residual:     
            y = {f"tau{i}_minus_g": df[f"torque_cmd{i}"] - df[f"tau_g{i}"] for i in range(7)}      #torque with gravity removed
        else:
            y = {f"torque_cmd{i}": df[f"torque_cmd{i}"] for i in range(7)}              #the expert’s torques, or only the steering part with gravity removed.
    return pd.DataFrame(X, index=df.index), pd.DataFrame(y, index=df.index)

def build_dataset(name, model="arm", stride=10, residual=False):
    folder = os.path.join(DATA_DIR, name)

    reasons = pd.read_csv(os.path.join(folder, "summary.csv")).drop_duplicates(             #lookup episode id to its outcome
        "episode_id", keep="last").set_index("episode_id")["stop_reason"]
    
    Xs, ys, ids, phases = [], [], [], []   

    for f in sorted(glob.glob(os.path.join(folder, "ep*.parquet"))):     
        df = pd.read_parquet(f)
        eid = int(df["episode_id"].iloc[0])
        if reasons.get(eid) in ("unreachable", "step_cap"):                #Only failed episode gets trimmed
            err = np.linalg.norm(df[["ee_x", "ee_y", "ee_z"]].to_numpy()    #dinstace from hand to target
                                 - df[["target_x", "target_y", "target_z"]].to_numpy(), axis=1)
            df = df.iloc[: int(np.argmax(err <= err.min() + 0.002)) + 1]        #finds first case where error is smallest
        reaching = df["nav_done"].astype(bool)                                  #when base parked = true
        df = (df[~reaching] if model == "base" else df[reaching]).iloc[::stride]            #base model learns from driving rows, the arm model from reaching rows
        if len(df) == 0:
            continue
        X, y = make_features(df, model, residual)  #turns into input/labels
        Xs.append(X)
        ys.append(y)
        ids.append(np.full(len(df), eid))
        phases.append(df["phase"].astype(str).to_numpy())
 
    X = pd.concat(Xs, ignore_index=True)   
    y = pd.concat(ys, ignore_index=True)
    assert not X.isna().any().any() and not y.isna().any().any(), "NaNs in dataset"
    out = os.path.join(folder, f"dataset_{model}{'_residual' if residual else ''}.npz")
    np.savez_compressed(out, X=X.to_numpy(np.float32), y=y.to_numpy(np.float32),
                        episode_id=np.concatenate(ids), phase=np.concatenate(phases).astype(str),
                        feature_names=np.array(X.columns, dtype=str), label_names=np.array(y.columns, dtype=str))
    print(f"saved {out}: X {X.shape}, y {y.shape}, {len(Xs)} episodes")    #saves episodes if completed

#running the program------------------------------

STEP = "dataset"

POOLS = {
    #"test":     (20,  "train",     0.0),
    "main":      (420, "train",     0.0),
    "oor_far":   (30,  "oor_far",   0.0),
    "oor_high":  (30,  "oor_high",  0.0),
    "oor_start": (30,  "oor_start", 0.0),
    "oor_arm":   (30,  "oor_arm",   0.0),
    "dart":      (150, "train",     0.5),
}

METHOD, WORKERS, LIMIT = "lhs", 8, None
MODEL, STRIDE, RESIDUAL = "base", 10, False

if __name__ == "__main__":       # still REQUIRED: worker processes re-import this file
    for name, (n, ranges, noise) in POOLS.items():
        if STEP == "plan":
            path = os.path.join(DATA_DIR, name, "plan.csv")
            if os.path.exists(path):
                raise SystemExit(f"{path} exists - delete it first if nothing is collected")
            others = [pd.read_csv(f)["episode_id"].max() for f in glob.glob(os.path.join(DATA_DIR, "*", "plan.csv"))]
            first_id = int(max(others, default=-1)) + 1
            plan = make_plan(name, n, first_id, first_id, OOR[ranges], METHOD, noise)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plan.to_csv(path, index=False)
            plot_coverage(plan, os.path.join(OUT_DIR, f"plan_{name}.png"), f"{name}: {METHOD}, n={n}, {ranges}")
        elif STEP == "collect":
            collect_data(name, WORKERS, LIMIT)
        elif STEP == "qa":
            qa(name)
        elif STEP == "dataset":
            build_dataset(name, MODEL, STRIDE, RESIDUAL)
