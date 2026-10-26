import numpy as np
import torch
import torch.nn as nn
from features import make_features


def make_mlp():
    # MUST be identical to the model you trained (same layers, same order, same dropout).
    return nn.Sequential(
        nn.Linear(32, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 11),
    )


class MLPPolicy:
    def __init__(self, model_path="model_best.pt", data_path="data/dataset.npz"):
        d = np.load(data_path)
        self.x_mean, self.x_std = d["x_mean"], d["x_std"]
        self.y_mean, self.y_std = d["y_mean"], d["y_std"]
        self.model = make_mlp()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        torch.set_num_threads(1)          # one small row per call: threads only add overhead

    def __call__(self, base_pos, base_bearing, q_mes, qd_mes, ee_pos, target, t):
        # Pack ONE time step as arrays of length 1, so make_features works unchanged
        d = {
            "base_pos": base_pos[None, :],
            "base_bearing": np.array([base_bearing]),
            "q_mes": q_mes[None, :],
            "qd_mes": qd_mes[None, :],
            "ee_pos": ee_pos[None, :],
            "target": np.asarray(target),
            "time": np.array([t]),
        }
        x = (make_features(d) - self.x_mean) / self.x_std
        with torch.no_grad():
            y_n = self.model(torch.tensor(x, dtype=torch.float32)).numpy()[0]
        y = y_n * self.y_std + self.y_mean
        return y[:4], y[4:11]             # wheel_cmds (4), torque_arm (7)