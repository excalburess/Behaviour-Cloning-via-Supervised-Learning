"""GRU (recurrent) policy on top of Noah's pipeline. Noah's files are not edited.

Why a GRU: the expert has memory the MLP cannot see. The base controller is in one of several phases
(align / drive / orient) and the arm controller integrates its own joint reference q_des over time (clipped to
0.15 rad ahead of q). A GRU carries a hidden state through the episode, so it can infer that history from what it
has seen so far. Same split as Noah's Policy: the base GRU drives until the base is parked (nav_done), the arm GRU
reaches after that.

    e_t = SiLU(Linear(x_t))          feature embedding
    h_k = GRU(e_k, h_{k-1})          memory, updated every 10 calls (10 ms, the STRIDE the sequences are built with)
    y_t = MLP([h_k, e_t])            head: memory + current features -> command, every call (1 ms)

Pick a STEP at the bottom, then: pixi run python gru.py
  "pools"       : plans and collects the pools pipeline.py does not define (dart2, sel), see EXTRA_POOLS
  "train"       : trains the base and arm GRUs on main's train split + EXTRA pools, saves models/<NAME>.joblib
                  (with the train/val loss of every epoch, for Fig. 2)
  "ensemble"    : one model file whose arm is the mean of several trained arm GRUs (MEMBERS)
  "closed_loop" : drives the robot with models/<NAME>.joblib on main's SPLIT or another POOL (oor_*, sel), optionally
                  with sensor noise / actuator noise / low friction; prints success and final error mean +- std
  "dagger"      : the GRU drives on main's train episodes while the expert labels in shadow (DART noise optional);
                  the logs are saved as a new pool data/<POOL>/ that "train" can use as an EXTRA pool
  "report"      : parameters, file size, inference time, offline test NMSE, closed-loop numbers -> gru_report.csv,
                  and the loss curves of the last REPORT model -> gru_loss_curves_<NAME>.png
Inference is plain numpy (no torch in the simulator processes).

Final model (models/gru_e80.joblib), after pipeline.py has collected main + dart and "pools" has collected dart2:
  1. NAME = "gru_dart2", EXTRA = ["dart", "dart2"], EPOCHS = 40, BASE_FROM = None       -> "train"
  2. NAME = "gru_e80",   EXTRA = ["dart", "dart2"], EPOCHS = 80, BASE_FROM = "gru_dart2" -> "train" (arm only)
Chosen on val + sel (200 episodes) among seeds / epochs / near-target weighting / a 3-seed ensemble; test used once.
DAgger did not help: the expert's labels depend on its own q_des integrator, which winds up to its 0.15 rad clip
while the GRU drives, so the logged torques are one long saturated push (gru_dagger1: 71.5% on sel vs 92.7%).
"""
import os
import glob
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from robotnik_arm_task import STANDOFF_Z_POINTS, STANDOFF_POINTS, ARM_REACH_LIMIT, TORQUE_LIMIT
from training import load_splits, MODEL_DIR, OUT_DIR, DATA_DIR

STRIDE = 10             #memory update period in calls = row stride of the training sequences
MAX_WHEEL = 3.5         #the expert never goes above 3 rad/s

BASE_FEATURES = ["tgt_rel_x", "tgt_rel_y", "tgt_z", "goal_rel_x", "goal_rel_y", "goal_dist",
                 "goal_heading_sin", "goal_heading_cos", "final_bearing_err_sin", "final_bearing_err_cos"]
ARM_FEATURES = (["tgt_rel_x", "tgt_rel_y", "tgt_z"] + [f"q_sin{i}" for i in range(7)] + [f"q_cos{i}" for i in range(7)]
                + [f"q{i}" for i in range(7)] + [f"qd{i}" for i in range(7)]
                + ["ee_err_x", "ee_err_y", "ee_err_z", "ee_err_norm", "reach_err_x", "reach_err_y", "reach_err_z",
                   "reach_err_norm"])
BASE_LABELS = ["cmd_wheel0", "cmd_wheel1"]                        #right, left (the expert sends [r, l, l, r])
ARM_LABELS = [f"tau{i}_minus_g" for i in range(7)]                #torque minus gravity, tau_g added back at run time


# ---------------------------------------------------------------- features (same code for logged data and live state)
def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gru_inputs(d):
    #d: DataFrame of logged rows, or a dict of one live state (state dict of run_episode). Robot-frame geometry.
    g = lambda k: np.atleast_1d(np.asarray(d[k], dtype=np.float64))
    yaw, bx, by = g("base_yaw"), g("base_x"), g("base_y")
    tx, ty, tz = g("target_x"), g("target_y"), g("target_z")
    c, s = np.cos(yaw), np.sin(yaw)
    rot = lambda x, y: (c * x + s * y, -s * x + c * y)            #world -> robot frame
    X = {}
    X["tgt_rel_x"], X["tgt_rel_y"] = rot(tx - bx, ty - by)
    X["tgt_z"] = tz
    #parking pose, same rule as robotnik_arm_task.py
    standoff = np.interp(tz, STANDOFF_Z_POINTS, STANDOFF_POINTS)
    goal_yaw = np.where(np.hypot(tx, ty) > 1e-4, np.arctan2(ty, tx), 0.0)
    gx, gy = tx - standoff * np.cos(goal_yaw), ty - standoff * np.sin(goal_yaw)
    X["goal_rel_x"], X["goal_rel_y"] = rot(gx - bx, gy - by)
    X["goal_dist"] = np.hypot(X["goal_rel_x"], X["goal_rel_y"])
    heading = np.arctan2(X["goal_rel_y"], X["goal_rel_x"])
    X["goal_heading_sin"], X["goal_heading_cos"] = np.sin(heading), np.cos(heading)
    final_err = wrap(goal_yaw - yaw)
    X["final_bearing_err_sin"], X["final_bearing_err_cos"] = np.sin(final_err), np.cos(final_err)
    q = np.stack([g(f"q{i}") for i in range(7)], axis=1)
    for i in range(7):
        X[f"q_sin{i}"], X[f"q_cos{i}"] = np.sin(q[:, i]), np.cos(q[:, i])
        X[f"q{i}"], X[f"qd{i}"] = q[:, i], g(f"qd{i}")
    #hand -> target, and the reach-limited version the expert's IK chases (far targets projected onto the sphere)
    ee = np.stack([g("ee_x"), g("ee_y"), g("ee_z")], axis=1)
    sh = np.stack([g("shoulder_x"), g("shoulder_y"), g("shoulder_z")], axis=1)
    tgt = np.stack([tx, ty, tz], axis=1)
    off = tgt - sh
    n = np.linalg.norm(off, axis=1, keepdims=True)
    off = np.where(n > ARM_REACH_LIMIT, off * ARM_REACH_LIMIT / np.maximum(n, 1e-9), off)
    for name, err in (("ee_err", tgt - ee), ("reach_err", sh + off - ee)):
        X[f"{name}_x"], X[f"{name}_y"] = rot(err[:, 0], err[:, 1])
        X[f"{name}_z"] = err[:, 2]
        X[f"{name}_norm"] = np.linalg.norm(err, axis=1)
    return X


def gru_labels(df):
    y = {"cmd_wheel0": df["cmd_wheel0"].to_numpy(), "cmd_wheel1": df["cmd_wheel1"].to_numpy()}
    for i in range(7):
        y[f"tau{i}_minus_g"] = (df[f"torque_cmd{i}"] - df[f"tau_g{i}"]).to_numpy()
    return y


def load_sequences(pool, ids=None):
    #one (base, arm) pair of sequences per episode: driving rows / reaching rows, every STRIDE-th 1 ms row.
    #DAgger pools (name contains "dagger") are not trimmed: their stalls, labelled by the expert, are the point
    folder = os.path.join(DATA_DIR, pool)
    trim = "dagger" not in pool
    reasons = pd.read_csv(os.path.join(folder, "summary.csv")).drop_duplicates(
        "episode_id", keep="last").set_index("episode_id")["stop_reason"]
    base, arm = [], []
    for f in sorted(glob.glob(os.path.join(folder, "ep*.parquet"))):
        eid = int(os.path.basename(f)[2:7])
        if (ids is not None and eid not in ids) or reasons.get(eid) in (None, "crash"):
            continue
        df = pd.read_parquet(f)
        if trim and reasons.get(eid) in ("unreachable", "step_cap"):  #trimmed like Noah's build_dataset
            err = np.linalg.norm(df[["ee_x", "ee_y", "ee_z"]].to_numpy()
                                 - df[["target_x", "target_y", "target_z"]].to_numpy(), axis=1)
            df = df.iloc[: int(np.argmax(err <= err.min() + 0.002)) + 1]
        reaching = df["nav_done"].astype(bool).to_numpy()
        for rows, out, xs, ys in ((df[~reaching], base, BASE_FEATURES, BASE_LABELS),
                                  (df[reaching], arm, ARM_FEATURES, ARM_LABELS)):
            rows = rows.iloc[::STRIDE]
            if len(rows) > 1:
                X, y = gru_inputs(rows), gru_labels(rows)
                out.append((np.stack([X[k] for k in xs], 1).astype(np.float32),
                            np.stack([y[k] for k in ys], 1).astype(np.float32),
                            (X["ee_err_norm"] if xs is ARM_FEATURES else np.zeros(len(rows))).astype(np.float32)))
    return base, arm


# ---------------------------------------------------------------- training (torch)
def build_net(n_in, n_out, hidden):
    import torch
    from torch import nn

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU())
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                      nn.Linear(hidden, n_out))

        def forward(self, x):
            e = self.emb(x)
            g, _ = self.gru(e)
            return self.head(torch.cat((g, e), dim=-1))

    return GRUNet()


def fit_gru(name, train, val, feature_names, label_names, hidden, epochs, batch, lr, near_weight, near_dist, seed=0):
    #train, val: lists of (X, y, ee_err_norm) sequences. Full episodes, zero initial memory (as at run time), padded
    #and masked; reaching rows within near_dist of the target weighted near_weight (the last centimetre matters most)
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_all, y_all = np.concatenate([x for x, _, _ in train]), np.concatenate([y for _, y, _ in train])
    x_mean, x_std = X_all.mean(0), X_all.std(0) + 1e-6
    y_mean, y_std = y_all.mean(0), y_all.std(0) + 1e-6

    def batches(seqs, idx):
        T = max(len(seqs[i][0]) for i in idx)
        X = np.zeros((len(idx), T, len(x_mean)), np.float32)
        Y = np.zeros((len(idx), T, len(y_mean)), np.float32)
        W = np.zeros((len(idx), T), np.float32)
        for b, i in enumerate(idx):
            x, y, near = seqs[i]
            X[b, :len(x)], Y[b, :len(x)] = (x - x_mean) / x_std, (y - y_mean) / y_std
            W[b, :len(x)] = np.where(near < near_dist, near_weight, 1.0) if near_weight != 1 else 1.0
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(W)

    def loss(net, X, Y, W):
        return (((net(X) - Y) ** 2).mean(-1) * W).sum(), W.sum()

    net = build_net(len(x_mean), len(y_mean), hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    order = np.argsort([len(s[0]) for s in train])                 #similar lengths per batch: little padding
    groups = [order[i:i + batch] for i in range(0, len(order), batch)]
    val_groups = [list(range(i, min(i + batch, len(val)))) for i in range(0, len(val), batch)]
    print(f"[{name}] {sum(p.numel() for p in net.parameters())} parameters, {len(train)} train / {len(val)} val episodes")
    best, best_state, history = np.inf, None, []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        tr_num, tr_den = 0.0, 0.0
        for gi in np.random.permutation(len(groups)):
            num, den = loss(net, *batches(train, groups[gi]))
            opt.zero_grad()
            (num / den).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tr_num, tr_den = tr_num + float(num), tr_den + float(den)
        sched.step()
        net.eval()
        with torch.no_grad():
            parts = [loss(net, *batches(val, g)) for g in val_groups]
            val_mse = float(sum(n for n, _ in parts) / sum(d for _, d in parts))
        if val_mse < best:                                         #keep the best epoch on the val split
            best, best_state = val_mse, {k: v.clone() for k, v in net.state_dict().items()}
        history.append({"epoch": epoch, "train_mse": tr_num / tr_den, "val_mse": val_mse})
        print(f"[{name}] epoch {epoch:3d} train {tr_num / tr_den:.4f} val {val_mse:.4f} (best {best:.4f}) "
              f"{time.time() - t0:.0f}s", flush=True)
    return {"weights": {k: v.numpy().astype(np.float64) for k, v in best_state.items()}, "hidden": hidden,
            "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
            "feature_names": feature_names, "label_names": label_names, "val_mse": best,
            "history": history, "train_s": time.time() - t_start,
            "n_params": int(sum(p.numel() for p in net.parameters()))}


def train(name, extra_pools, hidden=128, epochs=40, batch=8, lr=1e-3, near_weight=3.0, near_dist=0.05, seed=0,
          base_from=None):
    #base_from: name of a saved model whose base GRU is reused as is (DAgger rounds only retrain the arm)
    tr_ids, va_ids = set(load_splits("train")), set(load_splits("val"))
    base_tr, arm_tr = load_sequences("main", tr_ids)
    base_va, arm_va = load_sequences("main", va_ids)
    for pool in extra_pools:                                       #DART / DAgger pools: training only
        b, a = load_sequences(pool)
        base_tr, arm_tr = base_tr + b, arm_tr + a
    base = (joblib.load(os.path.join(MODEL_DIR, f"{base_from}.joblib"))["base"] if base_from else
            fit_gru("base", base_tr, base_va, BASE_FEATURES, BASE_LABELS, hidden, epochs, batch, lr, 1.0, 0, seed))
    model = {"kind": "gru", "stride": STRIDE, "base": base,
             "arm": fit_gru("arm", arm_tr, arm_va, ARM_FEATURES, ARM_LABELS, hidden, epochs, batch, lr,
                            near_weight, near_dist, seed),
             "extra_pools": list(extra_pools), "base_from": base_from}
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"saved {path}")


def ensemble(name, members):
    #one model file whose arm is the mean of the members' arm GRUs (base GRU from the first member)
    ms = [joblib.load(os.path.join(MODEL_DIR, f"{n}.joblib")) for n in members]
    model = {**ms[0], "arm_members": [m["arm"] for m in ms], "members": list(members)}
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"saved {path}")


# ---------------------------------------------------------------- policy (numpy)
def _silu(h):
    return h * 0.5 * (1.0 + np.tanh(0.5 * h))                      #overflow-free h * sigmoid(h)


def _sigmoid(h):
    return 0.5 * (1.0 + np.tanh(0.5 * h))


class _GRUNumpy:
    def __init__(self, m):
        w, H = m["weights"], m["hidden"]
        self.emb = (w["emb.0.weight"], w["emb.0.bias"])
        self.Wi = [w["gru.weight_ih_l0"][k * H:(k + 1) * H] for k in range(3)]      #torch gate order: reset, update, new
        self.Wh = [w["gru.weight_hh_l0"][k * H:(k + 1) * H] for k in range(3)]
        self.bi = [w["gru.bias_ih_l0"][k * H:(k + 1) * H] for k in range(3)]
        self.bh = [w["gru.bias_hh_l0"][k * H:(k + 1) * H] for k in range(3)]
        self.head = [(w[f"head.{k}.weight"], w[f"head.{k}.bias"]) for k in (0, 2, 4)]
        self.m, self.h = m, np.zeros(H)

    def __call__(self, x, update):
        e = _silu(self.emb[0] @ ((x - self.m["x_mean"]) / self.m["x_std"]) + self.emb[1])
        if update:
            r = _sigmoid(self.Wi[0] @ e + self.bi[0] + self.Wh[0] @ self.h + self.bh[0])
            z = _sigmoid(self.Wi[1] @ e + self.bi[1] + self.Wh[1] @ self.h + self.bh[1])
            n = np.tanh(self.Wi[2] @ e + self.bi[2] + r * (self.Wh[2] @ self.h + self.bh[2]))
            self.h = (1.0 - z) * n + z * self.h
        h = np.concatenate((self.h, e))
        for W, b in self.head[:-1]:
            h = _silu(W @ h + b)
        return (self.head[-1][0] @ h + self.head[-1][1]) * self.m["y_std"] + self.m["y_mean"]


class _Ensemble:
    #several arm GRUs (different training seeds), each with its own memory; the command is their mean
    def __init__(self, members):
        self.nets = [_GRUNumpy(a) for a in members]

    def __call__(self, x, update):
        return np.mean([net(x, update) for net in self.nets], axis=0)


class GRUPolicy:
    #same interface as Noah's Policy (closed_loop.py). One instance per episode: the memory starts empty.
    def __init__(self, model_path, drives_base=True, drives_arm=True):
        m = joblib.load(model_path)
        self.stride = m["stride"]
        self.base = _GRUNumpy(m["base"])
        self.arm = _Ensemble(m["arm_members"]) if m.get("arm_members") else _GRUNumpy(m["arm"])
        self.drives_base, self.drives_arm = drives_base, drives_arm
        self.base_calls, self.arm_calls = 0, 0

    def wheel_cmds(self, state):
        X = gru_inputs(state)
        x = np.array([X[k][0] for k in BASE_FEATURES])
        right, left = np.clip(self.base(x, self.base_calls % self.stride == 0), -MAX_WHEEL, MAX_WHEEL)
        self.base_calls += 1
        return np.array([right, left, left, right])

    def arm_torques(self, state):
        X = gru_inputs(state)
        x = np.array([X[k][0] for k in ARM_FEATURES])
        tau = self.arm(x, self.arm_calls % self.stride == 0) + np.array([state[f"tau_g{i}"] for i in range(7)])
        self.arm_calls += 1
        return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


class SensorNoise:
    #robustness test, same as Finn's (mlp_eval.py): the policy sees noisy joint readings (std q_std on q, 5x on qd);
    #the simulator and the expert still use the true state
    def __init__(self, policy, q_std, seed=0):
        self.policy, self.q_std, self.qd_std = policy, q_std, 5 * q_std
        self.rng = np.random.default_rng(seed)
        self.drives_arm, self.drives_base = policy.drives_arm, policy.drives_base

    def noisy(self, state):
        state = dict(state)
        for i in range(7):
            state[f"q{i}"] += self.rng.normal(0, self.q_std)
            state[f"qd{i}"] += self.rng.normal(0, self.qd_std)
        return state

    def wheel_cmds(self, state):
        return self.policy.wheel_cmds(self.noisy(state))

    def arm_torques(self, state):
        return self.policy.arm_torques(self.noisy(state))


# ---------------------------------------------------------------- closed loop / DAgger
def run_one(job):
    row, model_path, drives_base, drives_arm, noise_std, save_folder, gui, sensor_noise, friction = job
    os.environ["ROBOENV_HEADLESS"] = "0" if gui else "1"
    eid = int(row["episode_id"])
    start = time.perf_counter()
    try:
        from robotnik_arm_task import run_episode
        policy = GRUPolicy(model_path, drives_base, drives_arm)
        if sensor_noise > 0:
            policy = SensorNoise(policy, sensor_noise, seed=eid)
        df, summary = run_episode(
            target=[row["target_x"], row["target_y"], row["target_z"]],
            start_xy=[row["start_x"], row["start_y"]], start_yaw=float(row["start_yaw"]),
            start_arm_offset=np.array([row[f"arm_offset{i}"] for i in range(7)], dtype=float),
            seed=eid, noise_seed=eid + 100000, noise_std=noise_std, floor_friction=friction, headless=not gui,
            policy=policy)
        if save_folder:
            df["episode_id"] = eid
            df.to_parquet(os.path.join(save_folder, f"ep{eid:05d}.parquet"), index=False)
    except Exception as e:
        summary = {"stop_reason": "crash", "error": f"{type(e).__name__}: {e}"}
    summary.update(episode_id=eid, d=row["d"], height=row["height"], wall_time_s=time.perf_counter() - start)
    return summary


def run_many(jobs, workers, gui):
    if gui or workers == 1:
        return [run_one(j) for j in tqdm(jobs)]
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        return list(tqdm(pool.imap_unordered(run_one, jobs), total=len(jobs)))


def closed_loop(name, split, workers, gui=False, drives_base=True, drives_arm=True, n_episodes=None, pool="main",
                sensor_noise=0.0, act_noise=0.0, friction=1.0, tag=""):
    #pool "main" = the SPLIT episodes of main; any other pool (oor_*, sel) = all its episodes.
    #sensor_noise / act_noise / friction: robustness conditions, same meaning as in Finn's mlp_eval.py
    plan = pd.read_csv(os.path.join(DATA_DIR, pool, "plan.csv"))
    if pool == "main":
        plan = plan[plan["episode_id"].isin(load_splits(split))]
    plan = plan.head(1) if gui else (plan.head(n_episodes) if n_episodes else plan)
    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
    run_name = (f"{name}_{split if pool == 'main' else pool}" + ("" if drives_base else "_arm")
                + ("" if drives_arm else "_base") + tag)
    print(f"{run_name}: {len(plan)} {split if pool == 'main' else pool} episodes | base: "
          f"{'GRU' if drives_base else 'expert'} | arm: {'GRU' if drives_arm else 'expert'} | sensor noise "
          f"{sensor_noise}, act noise {act_noise}, friction {friction}")
    jobs = [(row, model_path, drives_base, drives_arm, act_noise, None, gui, sensor_noise, friction)
            for row in plan.to_dict("records")]
    runs = pd.DataFrame(run_many(jobs, workers, gui))
    os.makedirs(OUT_DIR, exist_ok=True)
    runs.to_csv(os.path.join(OUT_DIR, f"closed_loop_{run_name}.csv"), index=False)
    #the expert's own run of the same episodes (clean conditions)
    expert = pd.read_csv(os.path.join(DATA_DIR, pool, "summary.csv")).drop_duplicates("episode_id", keep="last")
    expert = expert[expert["episode_id"].isin(runs["episode_id"])]
    table = []
    for who, r in (("expert", expert), (run_name, runs)):
        reached = r["stop_reason"] == "reached"
        table.append({"who": who, "episodes": len(r), "success": reached.mean(),
                      "final_error_cm_mean": 100 * r["final_ee_error"].mean(),
                      "final_error_cm_std": 100 * r["final_ee_error"].std(),
                      "mean_time_s_when_reached": r.loc[reached, "sim_duration"].mean(),
                      "crashes": int((r["stop_reason"] == "crash").sum())})
    print("\n" + pd.DataFrame(table).round(3).to_string(index=False))
    for tol in (0.01, 0.02, 0.05):
        print(f"final error < {tol * 100:.0f} cm: {(runs['final_ee_error'] < tol).mean():.0%}")
    return runs


def dagger(name, pool, workers, noise_std=0.0):
    #the GRU drives on main's train episodes; the step logs hold the expert's commands for every state it visited
    folder = os.path.join(DATA_DIR, pool)
    os.makedirs(folder, exist_ok=True)
    plan = pd.read_csv(os.path.join(DATA_DIR, "main", "plan.csv"))
    plan = plan[plan["episode_id"].isin(load_splits("train"))]
    plan = plan[[not os.path.exists(os.path.join(folder, f"ep{i:05d}.parquet")) for i in plan["episode_id"]]]
    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
    jobs = [(row, model_path, True, True, noise_std, folder, False, 0.0, 1.0) for row in plan.to_dict("records")]
    print(f"[{pool}] {name} drives {len(jobs)} train episodes, noise_std {noise_std}")
    summary_csv = os.path.join(folder, "summary.csv")
    old = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else pd.DataFrame()
    pd.concat([old, pd.DataFrame(run_many(jobs, workers, False))]).to_csv(summary_csv, index=False)


# ---------------------------------------------------------------- extra pools
#pools that are not in pipeline.py's POOLS: (name, first episode id, episodes, ranges, noise_std). IDs continue
#Noah's numbering (test 0-19, main 20-439, oor_* 440-559, dart 560-709), seed = first id as in pipeline.py
EXTRA_POOLS = [("dart2", 710, 500, "train", 0.5),     #second DART pool (as Finn's plan_dart2.png; noise inferred)
               ("sel", 1210, 137, "train", 0.0)]      #extra selection episodes: with val, 200 for choosing models


def make_pools(workers):
    #plans (with Noah's make_plan) and collects the EXTRA_POOLS; resumable like pipeline.py's collect
    import pipeline
    for name, first, n, ranges, noise in EXTRA_POOLS:
        path = os.path.join(DATA_DIR, name, "plan.csv")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pipeline.make_plan(name, n, first, first, pipeline.OOR[ranges], "lhs", noise).to_csv(path, index=False)
        pipeline.collect_data(name, workers)


# ---------------------------------------------------------------- report (Table III numbers, Fig. 2 curves)
def offline_mse(net_models, seqs):
    #standardised MSE on held-out sequences (= NMSE: labels are scaled by the train std), memory reset per episode.
    #net_models: one network, or the members of an ensemble (prediction = their mean)
    num, den = 0.0, 0
    for X, Y, _ in seqs:
        net = _Ensemble(net_models)
        pred = np.array([net(x.astype(np.float64), True) for x in X])
        num += float((((pred - Y) / net_models[0]["y_std"]) ** 2).mean(1).sum())
        den += len(X)
    return num / den


def inference_us(model_path, n=2000):
    #time per control call on logged test states, features included (the robot calls the policy every 1000 us)
    eid = int(load_splits("test")[0])
    df = pd.read_parquet(os.path.join(DATA_DIR, "main", f"ep{eid:05d}.parquet"))
    out = {}
    for part, rows in (("base", df[~df["nav_done"].astype(bool)]), ("arm", df[df["nav_done"].astype(bool)])):
        states = rows.head(n).to_dict("records")
        policy = GRUPolicy(model_path)
        call = policy.wheel_cmds if part == "base" else policy.arm_torques
        t0 = time.perf_counter()
        for s in states:
            call(s)
        out[part] = 1e6 * (time.perf_counter() - t0) / len(states)
    return out


def report(rows):
    #rows: (model name, val run, test run). Prints and saves outputs/gru_report.csv
    test_ids = set(load_splits("test"))
    base_te, arm_te = load_sequences("main", test_ids)
    table = []
    for name, val_run, test_run in rows:
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        m = joblib.load(path)
        us = inference_us(path)
        arms = m.get("arm_members") or [m["arm"]]
        r = {"model": name, "params_base": sum(w.size for w in m["base"]["weights"].values()),
             "params_arm": sum(w.size for a in arms for w in a["weights"].values()),
             "file_kB": round(os.path.getsize(path) / 1024),
             "train_s_base": m["base"].get("train_s"), "train_s_arm": sum(a.get("train_s") or 0 for a in arms),
             "us_per_base_step": us["base"], "us_per_arm_step": us["arm"],
             "test_nmse_base": offline_mse([m["base"]], base_te), "test_nmse_arm": offline_mse(arms, arm_te)}
        for split, run in (("val", val_run), ("test", test_run)):
            p = os.path.join(OUT_DIR, f"closed_loop_{run}.csv")
            if run and os.path.exists(p):
                c = pd.read_csv(p)
                r[f"{split}_success"] = (c["stop_reason"] == "reached").mean()
                r[f"{split}_err_cm_mean"] = 100 * c["final_ee_error"].mean()
                r[f"{split}_err_cm_std"] = 100 * c["final_ee_error"].std()
        table.append(r)
    t = pd.DataFrame(table)
    print(t.round(4).T.to_string())
    t.to_csv(os.path.join(OUT_DIR, "gru_report.csv"), index=False)
    print("saved outputs/gru_report.csv")


def plot_curves(name):
    #Fig. 2: train and val loss per epoch, base and arm GRU (standardised MSE; arm train loss is near-weighted)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = joblib.load(os.path.join(MODEL_DIR, f"{name}.joblib"))
    parts = [p for p in ("base", "arm") if m[p].get("history")]
    fig, axes = plt.subplots(1, len(parts), figsize=(5.5 * len(parts), 3.6), squeeze=False)
    for ax, part in zip(axes[0], parts):
        h = pd.DataFrame(m[part]["history"])
        for col, colour, label in (("train_mse", "#2a78d6", "train"), ("val_mse", "#eb6834", "val")):
            ax.plot(h["epoch"], h[col], color=colour, lw=2, label=label)
            ax.annotate(label, (h["epoch"].iloc[-1], h[col].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color="#52514e")
        best = h.loc[h["val_mse"].idxmin()]
        ax.plot(best["epoch"], best["val_mse"], "o", ms=8, color="#eb6834", mec="#fcfcfb", mew=2)
        ax.annotate(f"kept: epoch {int(best['epoch'])}", (best["epoch"], best["val_mse"]), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9, color="#52514e")
        ax.set_yscale("log")
        ax.set_title(f"{part} GRU", fontsize=11, color="#0b0b0b")
        ax.set_xlabel("epoch", color="#52514e")
        ax.set_ylabel("MSE (standardised labels)", color="#52514e")
        ax.grid(True, which="major", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
        ax.set_xlim(0, h["epoch"].max() + 4)
    fig.suptitle(f"{name}: training and validation loss", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"gru_loss_curves_{name}.png")
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    print(f"saved {out}")


# ---------------------------------------------------------------- settings
STEP = "closed_loop"
NAME = "gru_e80"                    #models/<NAME>.joblib (final model)
EXTRA = ["dart", "dart2"]           #extra training pools, e.g. ["dart", "dart2", "gru_dagger1"]
EPOCHS = 80                         #train: 40 for the base (gru_dart2), 80 for the final arm
BASE_FROM = "gru_dart2"             #train: reuse the base GRU of this saved model (None = train the base too)
SPLIT = "val"                       #closed_loop: "val" while developing, "test" only for final results
POOL = "main"                       #closed_loop: "main" (SPLIT episodes), "sel" (extra selection episodes) or oor_*
SENSOR_NOISE, ACT_NOISE, FRICTION = 0.0, 0.0, 1.0   #closed_loop robustness conditions (as in mlp_eval.py)
WORKERS = 8
GUI = False
DAGGER_POOL, DAGGER_NOISE = "gru_dagger1", 0.0      #dagger: new pool name; noise_std > 0 makes it a DART pool
MEMBERS = ["gru_e80", "gru_e80_s1", "gru_e80_s2"]    #ensemble: models/<NAME>.joblib = mean of these arm GRUs
REPORT = [("gru_dart2", "gru_dart2_val", "gru_dart2_test"),
          ("gru_e80", "gru_e80_val", "gru_e80_test")]                 #report: (model, val run, test run)

if __name__ == "__main__":
    if STEP == "train":
        train(NAME, EXTRA, epochs=EPOCHS, base_from=BASE_FROM)
    elif STEP == "closed_loop":
        closed_loop(NAME, SPLIT, WORKERS, GUI, pool=POOL, sensor_noise=SENSOR_NOISE, act_noise=ACT_NOISE,
                    friction=FRICTION)
    elif STEP == "dagger":
        dagger(NAME, DAGGER_POOL, WORKERS, DAGGER_NOISE)
    elif STEP == "pools":
        make_pools(WORKERS)
    elif STEP == "ensemble":
        ensemble(NAME, MEMBERS)
    elif STEP == "report":
        report(REPORT)
        plot_curves(REPORT[-1][0])
