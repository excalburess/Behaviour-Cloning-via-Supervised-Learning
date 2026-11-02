"""Torch MLP policy on top of Noah's pipeline: evaluation side. Noah's files are not edited.

Pick a STEP at the bottom, then: OMP_NUM_THREADS=1 pixi run python mlp_eval.py
  "closed_loop" : drives the robot with the models below (Noah's Policy, or EPolicy when QD_MODEL is set),
                  on main's SPLIT or on an oor_* pool, optionally with robustness adversities
  "analyse"     : failure analysis of closed-loop runs (success by target distance/height, failed episodes, png)
  "efficiency"  : parameters / file size / inference time / success table + png for the model sizes
"""
import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import mlp                                         #noqa: F401  lets the torch models load (TorchMLP)
from pipeline import make_inputs
from training import predict_physical, load_splits, MODEL_DIR, OUT_DIR, DATA_DIR, TORQUE_LIMIT
from closed_loop import Policy                     #Noah's policy (expert-style, no memory)


# ---------------------------------------------------------------- policy that tracks the expert's hidden reference
MAX_Q_LEAD = 0.15     #ArmParams.max_q_lead in robotnik_arm_control.py
DT = 0.001            #simulation step (1 ms): the arm model must run every step


class EPolicy(Policy):
    #keeps its own copy of the expert's reference lead e = q_des - q, updated like the expert (robotnik_arm_control.py):
    #q_des <- clip(q_des + qd_des * dt, q - MAX_Q_LEAD, q + MAX_Q_LEAD); qd_des from a learnt model ("qddes"),
    #the torque from a model that also sees e ("arm_e")
    def __init__(self, arm_model, qd_model, base_model=None):
        super().__init__(arm_model, base_model)
        self.qd_model = joblib.load(qd_model)
        self.e, self.q_prev = None, None

    @staticmethod
    def predict_with(trained_model, state, extra):
        X = make_inputs(pd.DataFrame([state]), "arm")
        for k, v in extra.items():                                    #adds e0..e6, which make_inputs can't compute
            X[k] = v
        X = X[trained_model["feature_names"]].to_numpy(dtype=np.float64)
        return predict_physical(trained_model, X)[0]

    def arm_torques(self, state):
        q = np.array([state[f"q{i}"] for i in range(7)])
        if self.e is None:                                            #first reaching step: the reference starts at the held pose
            self.e, self.q_prev = np.zeros(7), q
        qd_des = self.predict_with(self.qd_model, state, {})
        #q_des moves by qd_des*dt, q moved by (q - q_prev) since last step, then the anti-windup clip
        self.e = np.clip(self.e + qd_des * DT - (q - self.q_prev), -MAX_Q_LEAD, MAX_Q_LEAD)
        self.q_prev = q
        torques = self.predict_with(self.arm, state, {f"e{i}": self.e[i] for i in range(7)})
        self.last_torques = np.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)
        self.arm_calls += 1
        return self.last_torques


class SensorNoise:
    #robustness test: the policy sees noisy joint readings (the simulator and expert still use the true state)
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


# ---------------------------------------------------------------- closed loop
def run_one(job):
    row, gui = job
    os.environ["ROBOENV_HEADLESS"] = "0" if gui else "1"
    episode_id = int(row["episode_id"])
    start = time.perf_counter()
    try:
        from robotnik_arm_task import run_episode
        policy = EPolicy(ARM_MODEL, QD_MODEL, BASE_MODEL) if QD_MODEL else Policy(ARM_MODEL, BASE_MODEL)
        if SENSOR_NOISE > 0:
            policy = SensorNoise(policy, SENSOR_NOISE, seed=episode_id)
        df, summary = run_episode(
            target=[row["target_x"], row["target_y"], row["target_z"]],
            start_xy=[row["start_x"], row["start_y"]], start_yaw=float(row["start_yaw"]),
            start_arm_offset=np.array([row[f"arm_offset{i}"] for i in range(7)], dtype=float),
            seed=episode_id, noise_seed=episode_id + 100000, noise_std=ACT_NOISE, floor_friction=FRICTION,
            headless=not gui, policy=policy)
        if SAVE_STEPS:
            folder = os.path.join(OUT_DIR, "closed_loop", NAME)
            os.makedirs(folder, exist_ok=True)
            df.to_parquet(os.path.join(folder, f"ep{episode_id:05d}.parquet"), index=False)
    except Exception as e:
        summary = {"stop_reason": "crash", "error": f"{type(e).__name__}: {e}"}
    summary.update(episode_id=episode_id, d=row["d"], height=row["height"], wall_time_s=time.perf_counter() - start)
    return summary


def compare_with_expert(model_runs):
    #the expert's own run of the same episodes (data/<POOL>/summary.csv: clean conditions)
    expert = (pd.read_csv(os.path.join(DATA_DIR, POOL, "summary.csv"),
                          usecols=["episode_id", "stop_reason", "final_ee_error", "sim_duration"])
              .drop_duplicates("episode_id", keep="last"))
    expert = expert[expert["episode_id"].isin(model_runs["episode_id"])]
    table = []
    for who, runs in (("expert", expert), (NAME, model_runs)):
        reached = runs["stop_reason"] == "reached"
        table.append({"who": who, "episodes": len(runs), "success": reached.mean(),
                      "median_final_error_m": runs["final_ee_error"].median() if "final_ee_error" in runs else np.nan,
                      "mean_time_s_when_reached": runs.loc[reached, "sim_duration"].mean() if reached.any() else np.nan,
                      "crashes": int((runs["stop_reason"] == "crash").sum())})
    print("\n" + pd.DataFrame(table).round(4).to_string(index=False))
    print("\nstop reasons:\n" + model_runs["stop_reason"].value_counts().to_string())
    for _, r in model_runs[model_runs["stop_reason"] == "crash"].head(3).iterrows():
        print(f"crash in episode {r['episode_id']}: {r.get('error')}")


def closed_loop():
    plan = pd.read_csv(os.path.join(DATA_DIR, POOL, "plan.csv"))
    if POOL == "main":
        plan = plan[plan["episode_id"].isin(load_splits(SPLIT))]
    if GUI:
        plan = plan.head(1)
    elif N_EPISODES:
        plan = plan.head(N_EPISODES)
    jobs = [(row, GUI) for row in plan.to_dict("records")]
    print(f"{NAME}: {len(jobs)} {SPLIT if POOL == 'main' else POOL} episodes | arm: {ARM_MODEL or 'expert'} | "
          f"qd: {QD_MODEL or '-'} | base: {BASE_MODEL or 'expert'}")
    if GUI or WORKERS == 1:
        results = [run_one(job) for job in tqdm(jobs)]
    else:
        with mp.get_context("spawn").Pool(WORKERS, maxtasksperchild=1) as pool:
            results = list(tqdm(pool.imap_unordered(run_one, jobs), total=len(jobs)))
    model_runs = pd.DataFrame(results)
    os.makedirs(OUT_DIR, exist_ok=True)
    model_runs.to_csv(os.path.join(OUT_DIR, f"closed_loop_{NAME}.csv"), index=False)
    compare_with_expert(model_runs)


# ---------------------------------------------------------------- failure analysis
def load_run(name):
    r = pd.read_csv(os.path.join(OUT_DIR, f"closed_loop_{name}.csv"))
    r["reached"] = r["stop_reason"] == "reached"
    return r


def analyse():
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for name in ANALYSE_RUNS:
        if not os.path.exists(os.path.join(OUT_DIR, f"closed_loop_{name}.csv")):
            print(f"skipping {name}: no csv")
            continue
        r = load_run(name)
        rows.append({"run": name, "episodes": len(r), "success": r["reached"].mean(),
                     "median_best_m": r["best_ee_error"].median(), "median_final_m": r["final_ee_error"].median(),
                     "got_within_2cm_then_failed": int(((r["best_ee_error"] < 0.02) & ~r["reached"]).sum())})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    r = load_run(FOCUS)
    for col in ("d", "height"):
        print(f"\nsuccess by {col}:\n" + r.groupby(pd.cut(r[col], 4), observed=True)["reached"].agg(["mean", "count"]).round(2).to_string())
    print(f"\nfailed episodes in {FOCUS}:\n" + r.loc[~r["reached"], ["episode_id", "d", "height", "stop_reason",
                                                                  "best_ee_error", "final_ee_error"]].round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for ok, colour, label in ((True, "tab:green", "reached"), (False, "tab:red", "failed")):
        s = r[r["reached"] == ok]
        ax[0].scatter(s["d"], s["height"], c=colour, label=f"{label} ({len(s)})", s=25)
    ax[0].set_xlabel("target distance d (m)"); ax[0].set_ylabel("target height (m)")
    ax[0].set_title(f"{FOCUS}: outcome by target"); ax[0].legend()
    files = {int(os.path.basename(f)[2:7]): f for f in glob.glob(os.path.join(OUT_DIR, "closed_loop", FOCUS, "ep*.parquet"))}
    failed = [e for e in r.loc[~r["reached"], "episode_id"] if e in files]
    for eid in failed:
        df = pd.read_parquet(files[eid])
        err = np.linalg.norm(df[["ee_x", "ee_y", "ee_z"]].to_numpy() - df[["target_x", "target_y", "target_z"]].to_numpy(), axis=1)
        ax[1].plot(df["t"], err, lw=1, label=f"ep {eid}")
    ax[1].axhline(0.01, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("hand to target (m)")
    ax[1].set_title("failed episodes" if failed else "failed episodes (no step logs: rerun with SAVE_STEPS = True)")
    if failed:
        ax[1].legend(fontsize=7)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"failures_{FOCUS}.png")
    plt.savefig(out, dpi=120)
    print(f"\nsaved {out}")


# ---------------------------------------------------------------- efficiency
def efficiency():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = pd.read_csv(os.path.join(OUT_DIR, "results_offline.csv")).drop_duplicates(["model", "dataset"], keep="last").set_index(["model", "dataset"])

    def success(name):
        p = os.path.join(OUT_DIR, f"closed_loop_{name}.csv")
        return (pd.read_csv(p).stop_reason == "reached").mean() if os.path.exists(p) else float("nan")

    rows = []
    for label, prefix, val, test in EFFICIENCY_MODELS:
        ds = ["arm_e", "qddes", "base"]
        rows.append({"model": label,
                     "params": int(sum(r.loc[(prefix, d), "n_params"] for d in ds)),
                     "files_kB": round(sum(os.path.getsize(os.path.join(MODEL_DIR, f"{prefix}_{d}.joblib")) for d in ds) / 1024),
                     "us_per_arm_step": round(r.loc[(prefix, "arm_e"), "infer_us"] + r.loc[(prefix, "qddes"), "infer_us"], 1),
                     "us_per_base_step": round(r.loc[(prefix, "base"), "infer_us"], 1),
                     "val_success": success(val), "test_success": success(test)})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    t.to_csv(os.path.join(OUT_DIR, "efficiency_table.csv"), index=False)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    x = list(range(len(t)))
    ax[0].bar(x, t["params"], color="tab:blue")
    ax[0].set_ylabel("parameters (all 3 nets)"); ax[0].set_title("model size")
    for i, p in enumerate(t["params"]):
        ax[0].text(i, p, f"{p / 1000:.0f}k", ha="center", va="bottom")
    ax[1].bar(x, t["us_per_arm_step"], color="tab:orange")
    ax[1].set_ylabel("µs per arm step (arm_e + qddes)"); ax[1].set_title("inference time")
    w = 0.38
    ax[2].bar([i - w / 2 for i in x], 100 * t["val_success"], w, label="val")
    ax[2].bar([i + w / 2 for i in x], 100 * t["test_success"], w, label="test")
    ax[2].set_ylim(0, 100); ax[2].set_ylabel("closed-loop success (%)"); ax[2].set_title("does it still work?")
    ax[2].legend(loc="lower right")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(t["model"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "efficiency.png"), dpi=150)
    print("saved outputs/efficiency.png and outputs/efficiency_table.csv")


# ---------------------------------------------------------------- settings
STEP = "closed_loop"

#closed_loop (final model: 128x2 EPolicy). Normal policy: QD_MODEL = None + an "arm" model. Expert: all three None
NAME = "small_e_h128"
ARM_MODEL = os.path.join(MODEL_DIR, "size_h128_l2_arm_e.joblib")
QD_MODEL = os.path.join(MODEL_DIR, "size_h128_l2_qddes.joblib")      #set = EPolicy, ARM_MODEL must be an arm_e model
BASE_MODEL = os.path.join(MODEL_DIR, "size_h128_l2_base.joblib")
POOL = "main"            #"main" = SPLIT episodes of main; oor_far / oor_high / oor_start / oor_arm = all their episodes
SPLIT = "val"
N_EPISODES = None
WORKERS = 7
GUI = False
SAVE_STEPS = False       #True keeps step logs (needed for DAgger and the failure curves)
ACT_NOISE = 0.0          #robustness: random kicks on the sent commands, DART scale (0.5 = training level)
FRICTION = 1.0           #robustness: floor friction (lower = slippery)
SENSOR_NOISE = 0.0       #robustness: std (rad) of noise on the joint angles the policy sees (velocities 5x)

#analyse
ANALYSE_RUNS = ["torch_mlp_both", "torch_mlp_dart_both", "torch_mlp_dart2_both", "torch_mlp_dart2_e_both", "small_e_h128"]
FOCUS = "torch_mlp_dart2_e_both"

#efficiency: (label, model prefix, val run NAME, test run NAME)
EFFICIENCY_MODELS = [("B 256x2", "torch_mlp_dart2", "torch_mlp_dart2_e_both", "test_dart2_e_both"),
                     ("128x2", "size_h128_l2", "small_e_h128", "test_small_e_h128"),
                     ("64x2", "size_h64_l2", "small_e_h64", "test_small_e_h64")]

if __name__ == "__main__":
    {"closed_loop": closed_loop, "analyse": analyse, "efficiency": efficiency}[STEP]()
