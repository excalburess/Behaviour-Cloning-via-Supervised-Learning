1) Make sure the folders "/models", "/configs" from RoboEnv are accessible (easiest is to copy them to this folder)
2) Use your RoboEnv environent (in whichever way you have been running the labs: pixi, uv, conda, etc)
3) Run robotnik_arm_task.py

Expected behaviour: pybullet 3D environment with 7dof arm mounted on a 4-wheeled platform. The robot moves to reach a target (red dot) with its end-effector. The simulation terminates when the target is reached, and plots of the robot behaviour are shown.

Key variables in robotnik_arm_task.py:

 - TARGET_EE_POS: reach target point in cartesian coordinates (x,y,z), represented by a red dot.

 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - torque_fingers: torque for each finger joint of the end-effector grasper. In this work, they should just remain fixed.
 
 Assessment Goal:
  - ML model that produces (wheel_cmds, torque_arm) from measurements (base_pos,base_bearing,q_mes,qd_mes)


## Notes made by Finn:
Finger torques wont be included in the data collectrion as they are not meanigful, they will be kept fixed. Parameters which will need to be measured each time step: 
 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - TARGET_EE_POS
 - Timestep

q_mes and qd_mes have 17 values. wheels[0:4], arm[4:11], fingers[11:17]. Wheel values grow as the wheels spin so this not useful, but the wheel speeds qd_mes are useful as these link to control parameters.

## Pipeline overview

| Stage | Script | Output | Status |
|---|---|---|---|
| 1. Run one expert episode | `robotnik_datacollect.py` | one `.npz` per run | done |
| 2. Collect many episodes | `collect.py` | `data/<split>/run_XXXX.npz` + `summary.json` | done |
| 3. Check the data | `check_data.py` | printed report + plots | TODO |
| 4. Build the training dataset | `build_dataset.py` | `data/dataset.npz` | TODO |
| 5. Train the model | `train.py` | model weights + loss curves | TODO |
| 6. Closed-loop evaluation | `evaluate.py` | reach error per test run | TODO |

---

## 1. Run a single expert episode

```
ROBOENV_HEADLESS=0 pixi run python robotnik_datacollect.py \
    --target 0.6 -0.89 1.02 \
    --base 0.27 -0.46 -2.88 \
    --arm-noise 0.09 \
    --seed 1 \
    --out data/test_episode.npz
```

| Argument | Meaning | Default |
|---|---|---|
| `--target x y z` | end-effector target in world frame (m) | `0.0 1.1 0.5` |
| `--base x y yaw` | starting base position (m) and heading (rad) | `0 0 0` |
| `--arm-noise std` | std (rad) of Gaussian noise added to the start arm joint angles | `0.0` |
| `--seed n` | seed for the arm-noise random generator | `0` |
| `--out path` | where to save the `.npz`; nothing is saved if omitted | none |

Environment variables:
- `ROBOENV_HEADLESS=1` disables the GUI and the plots, for fast data collection. `0` shows the GUI and the plots at the end.
- `ROBOENV_MAX_STEPS` caps the episode length in steps of 1 ms (default 40000).

The episode stops when the end-effector is within 10 mm of the target, when the reach stalls ("Target not reachable"), or at the step cap.

---

## 2. Collect a dataset

Edit the three constants at the top of `collect.py`, N_RUNS is the number of runs, MASTER_SEED is the seed for the rng numbers, so this can be switched for traing and testing data sets. OUT_DIR should be changed from data/train to data/test when ur creating test data. then run it:

```
run python collect.py
```



- Each run is launched as a separate headless process, so a crash or hang (600 s timeout) only loses that one run.
- Collection is **resumable**: runs whose `.npz` already exists are skipped, so you can stop it with Ctrl-C and rerun it, or raise `N_RUNS` to add more.
- Using the same `MASTER_SEED` regenerates exactly the same episodes. Different seeds keep the splits disjoint.

**Sampling (in `sample_episode`):**
- base start: x, y ~ U(-1, 1) m, yaw ~ U(-π, π)
- target: horizontal distance from the base ~ U(0.5, 3) m, direction relative to the base heading ~ U(-π, π), height ~ U(0.2, 1.1) m
- arm start noise std ~ U(0.001, 0.15) rad

Sampling the target relative to the base gives even coverage of relative target positions. Ranges are chosen with the arm reach limit (0.85 m from the shoulder at about 0.49 m height) and the standoff table in the template in mind.

### Contents of each `run_XXXX.npz`

Per time step (N rows, 1 ms apart):

| Key | Shape | Role |
|---|---|---|
| `time` | (N,) | elapsed time (s) |
| `base_pos` | (N, 3) | input |
| `base_bearing` | (N,) | input |
| `q_mes` | (N, 17) | input (use the arm part `[4:11]`; wheel angles `[0:4]` are not meaningful) |
| `qd_mes` | (N, 17) | input (arm `[4:11]`, wheel speeds `[0:4]`) |
| `ee_pos` | (N, 3) | candidate input (forward kinematics of the measured state) |
| `wheel_cmds` | (N, 4) | **label** |
| `torque_arm` | (N, 7) | **label** |
| `q_des_arm`, `qd_des_arm` | (N, 7) | expert internal state, for analysis only |
| `phase`, `done` | (N,) | expert base-navigation phase and the "arm reaching active" flag, for analysis only |

Per run: `target`, `start_base`, `seed`, `final_error` (m), `stop_message`.

`summary.json` lists every run with its sampled parameters and whether it ran and saved (`ok`). `ok = True` does not mean the target was reached. Check `stop_message` for that.

---


## External code and what was changed

- **Provided by the module (unchanged):** `robotnik_arm_control.py` (expert arm and base controllers, stop criteria), `robotnik_arm_task.py` (original expert template), `configs/`, `models/` (from RoboEnv), and the `simulation_and_control` library.
- **Modified from the template:** `robotnik_datacollect.py` is a copy of `robotnik_arm_task.py` with:
  - command-line arguments (`parse_args`) replacing the hard-coded `TARGET_EE_POS`
  - `set_start_pose()`, which resets the base pose and adds noise to the start arm joints after the simulation is created (only joints with finite limits are clipped)
  - extra per-step logging and saving to a compressed `.npz` before the headless early return
- **Written from scratch:** `collect.py` (random episode sampling and the collection loop).

