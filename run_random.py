"""Watch the GRU (or the expert) go for a randomly placed target (the red ball) in the pybullet window.

    pixi run python run_random.py

The episode is drawn the same way as the training data (pipeline.make_plan with the training ranges): target
0.8-2.5 m from the start in any direction, 0.15-1.10 m high, random start pose and arm offsets.
Set RANGES to one of the out-of-range sets for harder targets, POLICY = "expert" to watch the expert instead,
and SEED to an int to replay the same episode.
"""
import os
import time
import types
import numpy as np
import pipeline
import gru
import robotnik_arm_task
from robotnik_arm_task import run_episode

POLICY = "gru"          #"gru" = models/<MODEL>.joblib drives the base and arm, "expert" = the expert drives
MODEL = "gru_e80"
RANGES = "train"        #"train", "oor_far", "oor_high", "oor_start" or "oor_arm" (see pipeline.OOR)
SEED = None             #None = new random episode every run, an int = same episode every time
GUI = True              #False = no window, much quicker
FAST = True             #with the window open robotnik_arm_task.py sleeps 1 ms every step, so it runs at about
                        #real time. This skips the sleep. The episode itself comes out exactly the same

if __name__ == "__main__":
    if FAST:   #swap out time.sleep only inside robotnik_arm_task.py, so Noah's file doesn't need editing
        robotnik_arm_task.time = types.SimpleNamespace(**{k: getattr(time, k) for k in dir(time) if not k.startswith("_")})
        robotnik_arm_task.time.sleep = lambda s: None
    t0 = time.perf_counter()
    seed = int(np.random.default_rng().integers(1_000_000)) if SEED is None else SEED
    ep = pipeline.make_plan("random", 1, seed, seed, pipeline.OOR[RANGES], "random").iloc[0]
    print(f"\nseed {seed}: target ({ep.target_x:.2f}, {ep.target_y:.2f}, {ep.target_z:.2f}) m, "
          f"{ep.d:.2f} m from the start, start ({ep.start_x:.2f}, {ep.start_y:.2f}), heading {ep.start_yaw:.2f} rad")

    os.environ["ROBOENV_HEADLESS"] = "0" if GUI else "1"
    policy = gru.GRUPolicy(os.path.join(gru.MODEL_DIR, f"{MODEL}.joblib")) if POLICY == "gru" else None
    df, summary = run_episode(
        target=[ep.target_x, ep.target_y, ep.target_z], start_xy=[ep.start_x, ep.start_y], start_yaw=float(ep.start_yaw),
        start_arm_offset=np.array([ep[f"arm_offset{i}"] for i in range(7)], dtype=float),
        seed=seed, noise_seed=seed + 100000, headless=not GUI, policy=policy)
    print(f"\n{POLICY if POLICY == 'expert' else MODEL}: {summary['stop_reason']}, final error "
          f"{100 * summary['final_ee_error']:.2f} cm, {summary['sim_duration']:.1f} s simulated in "
          f"{time.perf_counter() - t0:.1f} s")
