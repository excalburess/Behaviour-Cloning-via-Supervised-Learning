import os, sys, subprocess, json
import numpy as np

N_RUNS = 50             # start small, then scale up
OUT_DIR = "data/test"
MASTER_SEED = 1         # use a different master seed for the test set

def sample_episode(rng):
    bx, by = (rng.uniform(-1.0,1.0,size=2))
    byaw = rng.uniform(-np.pi,np.pi)
    base = [float(bx), float(by), float(byaw)]

    dist = rng.uniform(0.5,3)
    angle = rng.uniform(-np.pi, np.pi)

    tz = rng.uniform(0.2,1.1)
    tx = bx + dist * np.cos(byaw+angle)
    ty = by + dist * np.sin(byaw + angle)
    target = (float(tx), float(ty), float(tz))
    armnoise = rng.uniform(0.001,0.15)
    dictionary = {"target":target,"base":base, "arm_noise":armnoise}
    print(dictionary)
    return dictionary
    

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(MASTER_SEED)
    env = dict(os.environ, ROBOENV_HEADLESS="1")
    summary = []
    for i in range(N_RUNS):
        ep = sample_episode(rng)
        seed = int(rng.integers(1_000_000))
        out = os.path.join(OUT_DIR, f"run_{i:04d}.npz")
        if os.path.exists(out):
            continue                       # lets you stop and resume collection
        cmd = [sys.executable, "robotnik_datacollect.py",
               "--target", *map(str, ep["target"]),
               "--base", *map(str, ep["base"]),
               "--arm-noise",  str(ep["arm_noise"]),
               "--seed", str(seed),
               "--out", out] 
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
            ok = result.returncode == 0 and os.path.exists(out)
        except subprocess.TimeoutExpired:
            ok = False
        summary.append({"run": i, "seed": seed, **ep, "ok": ok})
        print(i, ok)
        if not ok and 'result' in dir():
            print(result.returncode, result.stderr[-800:])
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()