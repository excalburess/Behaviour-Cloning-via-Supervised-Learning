import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from pipeline import make_inputs                      
from training import predict_physical, load_splits, MODEL_DIR, OUT_DIR, DATA_DIR, TORQUE_LIMIT


#Settings per model
NAME = "ridge_arm_every_1"
ARM_MODEL = os.path.join(MODEL_DIR, "ridge_arm.joblib")
BASE_MODEL = None
SPLIT = "val"
N_EPISODES = None
WORKERS = 3
GUI = True
SAVE_STEPS = True
POLICY_EVERY = 1 #rename



class Policy:
    def __init__(self, arm_model=None, base_model=None, every=POLICY_EVERY):
        self.arm = joblib.load(arm_model) if arm_model else None
        self.base = joblib.load(base_model) if base_model else None
        self.drives_arm = self.arm is not None
        self.drives_base = self.base is not None
        self.every = every                              #per how many calls do we measure
        self.arm_calls, self.base_calls = 0, 0
        self.last_torques, self.last_wheels = None, None

    @staticmethod                                                           #in class but independent of class objects
    def predict(trained_model, state, model_type):
        X = make_inputs(pd.DataFrame([state]), model_type)                   #computes arms model's features from raw readings
        X = X[trained_model["feature_names"]].to_numpy(dtype=np.float64)    #converts each feature to numpy
        
        return predict_physical(trained_model, X)[0]                        #runs model and unnormalizes scores

    def wheel_cmds(self, state):
        if self.base_calls % self.every == 0:
            out = dict(zip(self.base["label_names"], self.predict(self.base, state, "base")))
            right, left = out["cmd_right"], out["cmd_left"]
            self.last_wheels = np.array([right, left, left, right])
        self.base_calls += 1
        return self.last_wheels

    def arm_torques(self, state):
        if self.arm_calls % self.every == 0:
            torques = self.predict(self.arm, state, "arm")
            if self.arm["residual"]:
                torques = torques + np.array([state[f"tau_g{i}"] for i in range(7)])
            self.last_torques = np.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)
        self.arm_calls += 1
        return self.last_torques


def run_one(job):
    row, arm_model, base_model, gui = job = job   #unpacks job tuple 
    os.environ["ROBOENV_HEADLESS"] = "0" if gui else "1"
    episode_id = int(row["episode_id"])     #converts id into number
    start = time.perf_counter()
    try:
        #runs simulation
        from robotnik_arm_task import run_episode
        policy = Policy(arm_model, base_model)
        df, summary = run_episode(
            target=[row["target_x"], row["target_y"], row["target_z"]],
            start_xy=[row["start_x"], row["start_y"]], start_yaw=float(row["start_yaw"]),
            start_arm_offset=np.array([row[f"arm_offset{i}"] for i in range(7)], dtype=float),
            seed=episode_id, noise_std=0.0, headless=not gui, policy=policy)
        
        if SAVE_STEPS:
            #saves steps
            folder = os.path.join(OUT_DIR, "closed_loop", NAME)
            os.makedirs(folder, exist_ok=True)
            df.to_parquet(os.path.join(folder, f"ep{episode_id:05d}.parquet"), index=False)

    except Exception as e:
        summary: dict = {"stop_reason": "crash", "error": f"{type(e).__name__}: {e}"}
    summary.update(episode_id=episode_id, d=row["d"], height=row["height"],
                   wall_time_s=time.perf_counter() - start)
    return summary



def compare_with_expert(model_runs):
    expert = (pd.read_csv(os.path.join(DATA_DIR, "main", "summary.csv"),
                          usecols=["episode_id", "stop_reason", "final_ee_error", "sim_duration"])
                .drop_duplicates("episode_id", keep="last"))
    
    expert = expert[expert["episode_id"].isin(model_runs["episode_id"])]

    table = []
    for who, runs in (("expert", expert), (NAME, model_runs)):
        reached = runs["stop_reason"] == "reached"
        table.append({
            "who": who,
            "episodes": len(runs),
            "success": reached.mean(),
            "median_final_error_m": runs["final_ee_error"].median() if "final_ee_error" in runs else np.nan,
            "mean_time_s_when_reached": runs.loc[reached, "sim_duration"].mean() if reached.any() else np.nan,
            "crashes": int((runs["stop_reason"] == "crash").sum()),
        })
    print("\n" + pd.DataFrame(table).round(4).to_string(index=False))
    print("\nstop reasons:\n" + model_runs["stop_reason"].value_counts().to_string())
    for _, r in model_runs[model_runs["stop_reason"] == "crash"].head(3).iterrows():
        print(f"crash in episode {r['episode_id']}: {r.get('error')}")


if __name__ == "__main__":
    plan = pd.read_csv(os.path.join(DATA_DIR, "main", "plan.csv"))
    plan = plan[plan["episode_id"].isin(load_splits(SPLIT))] #checks if episodes in split of val and train and if so puts in plan
    if GUI:
        plan = plan.head(1)
    elif N_EPISODES:
        plan = plan.head(N_EPISODES)
    jobs = [(row, ARM_MODEL, BASE_MODEL, GUI) for row in plan.to_dict("records")] #turns job table into list of dict per episode
    #each job  = tuple holding everything ep needs for execution
    print(f"{NAME}: {len(jobs)} {SPLIT} episodes | arm: {ARM_MODEL or 'expert'} | base: {BASE_MODEL or 'expert'}")

    if GUI or WORKERS == 1:
        results = [run_one(job) for job in tqdm(jobs)]   #runs each job
    else: #runs in parallel other workers (spawn means brand new process)
        with mp.get_context("spawn").Pool(WORKERS, maxtasksperchild=1) as pool:
          #hands out the jobs and returns each summary as soon as episode finishes 
             results = list(tqdm(pool.imap_unordered(run_one, jobs), total=len(jobs)))

    model_runs = pd.DataFrame(results)
    os.makedirs(OUT_DIR, exist_ok=True)
    model_runs.to_csv(os.path.join(OUT_DIR, f"closed_loop_{NAME}.csv"), index=False)
    compare_with_expert(model_runs)