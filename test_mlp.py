import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pipeline import make_inputs
from training import predict_physical

bundle = joblib.load("models/torch_mlp_arm.joblib")
path = glob.glob("outputs/closed_loop/torch_mlp_arm/ep*.parquet")[0]
df = pd.read_parquet(path)
reach = df[df["nav_done"]]                                   # the steps where Ridge controlled the arm

X = make_inputs(reach, "arm")[bundle["feature_names"]].to_numpy(dtype=np.float64)
ridge_tau = np.clip(predict_physical(bundle, X), -30, 30)    # what Ridge data sent 
expert_tau = reach[[f"torque_cmd{i}" for i in range(7)]].to_numpy()   # what the expert would have sent
hand = reach[["ee_x", "ee_y", "ee_z"]].to_numpy()
goal = reach[["target_x", "target_y", "target_z"]].to_numpy()
err = np.linalg.norm(hand - goal, axis=1)
t = reach["t"].to_numpy()

print("first reaching step, expert:", np.round(expert_tau[0], 2))
print("first reaching step, ridge: ", np.round(ridge_tau[0], 2))

fig, ax = plt.subplots(8, 1, figsize=(10, 16), sharex=True)
ax[0].plot(t, err)
ax[0].set_ylabel("hand to target (m)")
for j in range(7):
    ax[j + 1].plot(t, expert_tau[:, j], label="expert would send")
    ax[j + 1].plot(t, ridge_tau[:, j], label="ridge sent")
    ax[j + 1].set_ylabel(f"joint {j} (N·m)")
ax[1].legend()
ax[-1].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("outputs/diagnose_torch_mlp_arm.png", dpi=120)
print("saved outputs/diagnose_torch_mlp_arm.png")