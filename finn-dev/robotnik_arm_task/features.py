import numpy as np
 
FEATURE_NAMES = (
    ["base_x", "base_y",
     "sin_heading", "cos_heading"]
    + [f"q_arm_{i}" for i in range(1, 8)]
    + [f"qd_arm_{i}" for i in range(1, 8)]
    + [f"wheel_speed_{i}" for i in range(1, 5)]
    + ["target_x", "target_y", "target_z"]
    + ["target_rel_x", "target_rel_y", "target_rel_z"]
    + ["ee_err_x", "ee_err_y", "ee_err_z"]
    + ["time"]
)
 
 
def make_features(d):
    base_x_y = d["base_pos"][:, :2]
    base_heading = d["base_bearing"]
    arm_angles = d["q_mes"][:, 4:11]
    arm_velocities = d["qd_mes"][:, 4:11]
    wheel_speeds = d["qd_mes"][:, 0:4]
    target = d["target"]
    ee_pos = d["ee_pos"]
    time = d["time"]
    n = len(time)
 
    # target relative to the base, rotated into the base frame
    target_rel_base = target[:2] - base_x_y
    c, s = np.cos(base_heading), np.sin(base_heading)
    dx, dy = target_rel_base[:, 0], target_rel_base[:, 1]
    trl = c * dx + s * dy
    trl2 = -s * dx + c * dy
    trl3 = np.full(n, target[2])
    target_rel = np.column_stack((trl, trl2, trl3))
 
    heading = np.column_stack((s, c))
    world_target = np.tile(target, (n, 1))
    ee_to_target_error = world_target - ee_pos
 
    list1 = [base_x_y, heading, arm_angles, arm_velocities, wheel_speeds,
             world_target, target_rel, ee_to_target_error, time[:, None]]
    X = np.concatenate(list1, axis=1)
    assert X.shape[1] == len(FEATURE_NAMES)
    return X
 