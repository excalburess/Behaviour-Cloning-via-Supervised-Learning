# Behaviour Cloning via Supervised Learning (COMP0245)


The projects is built to explore how different machine learning models can clone the behaviour of an expert controlling a robot, consisting of a driven base and an arm, to reach a target. A model learns to imitate the expert controller of a Robotnik mobile base with a 7-joint Kinova arm. The robot drives to a target (red dot) and reaches it with its hand.

---

## Contents

1. [Setup](#1-setup)
2. [The whole pipeline at a glance](#2-the-whole-pipeline-at-a-glance)
3. [Step 0: try the expert](#step-0-try-the-expert-optional)
4. [Steps 1–4: `pipeline.py` (plan, collect, qa, dataset)](#steps-14-pipelinepy)
5. [Step 5: check the datasets](#step-5-check-the-datasets-optional)
6. [Step 6: split the episodes](#step-6-split-the-episodes)
7. [Step 7: train the models](#step-7-train-the-models)
8. [Step 8: closed-loop test](#step-8-closed-loop-test)
9. [Step 9: diagnose a run](#step-9-diagnose-a-run-optional)
10. [GRU policy (`gru.py`)](#gru-policy-grupy)
11. [Output files](#output-files)
12. [External, modified, written from scratch](#external-modified-written-from-scratch)
13. [Original task notes](#original-task-notes)

---

## 1. Setup

1. Copy the `models/` and `configs/` folders from RoboEnv into this repo folder (robot description files).
2. Install the environment:
   ```
   pixi install
   ```
3. Run **every** script from the repo root like this:
   ```
   pixi run python <file>.py
   ```
4. Check you are in the right environment:
   ```
   pixi run python -c "import sys; print(sys.executable)"
   ```
   The path must be inside this repo's `.pixi` folder. If it shows `RoboEnv`, run `Remove-Item Env:PIXI_PROJECT_MANIFEST` in PowerShell, or open a new window.

---

## 2. The whole pipeline at a glance

Every step reads the previous step's files, so any step can be re-run on its own.
This origional pipeline was then reconstructed for the MLP and GRU models.

Origional Pipeline:

| Step | Script | What it does | Main output |
|---|---|---|---|
| 0 | `robotnik_arm_task.py` | Shows the expert (and a trained model) in the 3D window | PyBullet window |
| 1 | `pipeline.py`, `STEP = "plan"` | Chooses the targets and start poses of every episode | `data/<pool>/plan.csv` |
| 2 | `pipeline.py`, `STEP = "collect"` | Runs the expert on every planned episode | `data/<pool>/ep#####.parquet`, `summary.csv` |
| 3 | `pipeline.py`, `STEP = "qa"` | Checks the collected data | printed report + plot |
| 4 | `pipeline.py`, `STEP = "dataset"` | Turns raw logs into model inputs and labels | `data/<pool>/dataset_*.npz` |
| 5 | `check_datasets.py` | Checks two input-encoding choices | printed report |
| 6 | `splits.py` | Splits episodes into train / val / test | `splits/*_ids.txt` |
| 7 | `training.py` | Trains the Dummy and Ridge models, scores them | `models/*.joblib`, `outputs/results_offline.csv` |
| 8 | `closed_loop.py` | Lets a trained model drive the robot, compares with the expert | `outputs/closed_loop_<NAME>.csv` |
| 9 | `test.py` | Plots one closed-loop run against the expert | `outputs/diagnose_*.png` |

MLP Pipeline:

Step 7 and step 8 can be replaced with mlp.py and mlp_eval.py if an MLP is being trained and tested.
MLP uses a wrapper for a torch network so it can plug into policy and be used with the entire pipeline.
Settings can be chosen at the top for the hyper parameters of the MLP. This file also contains the Epolicy which is used for the hidden reference calculation. MLP eval can be used instead of close_loop to run test using the trained MLP models. Cluade Code was used as an assit to help with this part of the project.
---

## Step 0: try the expert (optional)

**Run:** `pixi run python robotnik_arm_task.py`

**What happens:** one episode runs twice in the PyBullet window: first the expert does everything, then the trained Ridge model controls the arm. A line per run is printed: outcome, final distance to the target, duration.

### `run_episode()` arguments

`run_episode()` in `robotnik_arm_task.py` runs one episode. Every other script calls it.

| Argument | Meaning |
|---|---|
| `target` | target position `[x, y, z]` in metres |
| `start_xy`, `start_yaw` | where the base starts, and which way it faces (rad) |
| `start_arm_offset` | 7 angle offsets (rad) added to the default arm pose; clipped to stay `joint_margin` inside the joint limits |
| `seed`, `noise_seed` | random seeds; the noise has its own seed so it never changes the episode's geometry |
| `noise_std` | DART noise level: 0 = none, 0.5 = noise of about half the normal steering effort |
| `headless` | `True` = no window (fast), `False` = show the 3D window |
| `floor_friction` | floor friction (1.0 normal; used for robustness tests) |
| `max_steps` | safety cap on steps (1 step = 1 ms) |
| `joint_margin` | how far inside the joint limits the start pose must stay (rad) |
| `verbose` | print extra information |
| `policy` | `None` = the expert controls everything; a `Policy` object = a trained model controls the arm and/or base |

It returns `df` (one row per 1 ms step) and `summary` (one row describing the episode).

### Constants at the top of `robotnik_arm_task.py`

| Constant | Meaning |
|---|---|
| `NOISE_HOLD_STEPS` | DART noise is redrawn every 150 steps and held in between |
| `WHEEL_NOISE_SCALE`, `TORQUE_NOISE_SCALE` | size of the noise per wheel / per joint at `noise_std = 1` |
| `TORQUE_LIMIT` | arm torques are clipped to ±30 N·m |
| `BASE_NAV_TIMEOUT` | after 25 s the base counts as parked, wherever it is |
| `STANDOFF_*` | how far before the target the base parks, depending on target height |
| `REACH_TOLERANCE` | success = hand within 1 cm of the target |

---

## Steps 1–4: `pipeline.py`

All four steps use the same file. Edit the **settings block at the bottom**, then run `pixi run python pipeline.py`.

### Settings

| Setting | Meaning | Typical value |
|---|---|---|
| `STEP` | which step to run: `"plan"`, `"collect"`, `"qa"` or `"dataset"` | change each time |
| `POOLS` | the groups of episodes to work on (see below). Comment out a line to skip that pool | all pools |
| `METHOD` | how targets are spread: `"lhs"` (Latin hypercube, even coverage) or `"random"` | `"lhs"` |
| `WORKERS` | how many simulators run in parallel during `collect` | 3–8 (fewer if RAM runs out) |
| `LIMIT` | only collect this many episodes (for a quick test); `None` = all | `None` |
| `MODEL` | dataset to build in the `dataset` step: `"arm"` or `"base"` | change each run |
| `STRIDE` | keep every N-th 1 ms row when building datasets (1 ms rows are near-copies) | `10` |
| `RESIDUAL` | arm only: label = torque minus gravity instead of full torque | `False` / `True` |

Each line of `POOLS` reads `"name": (episodes, ranges, noise_std)`:

| Pool | Episodes | Ranges | Noise | Used for |
|---|---|---|---|---|
| `main` | 420 | normal training ranges | 0 | train / val / test |
| `dart` | 150 | normal | 0.5 | extra training data with recovery examples |
| `oor_far` | 30 | targets 2.5–3.5 m away | 0 | test only: further than training |
| `oor_high` | 30 | targets 1.10–1.30 m high | 0 | test only: higher than training |
| `oor_start` | 30 | start positions 1.5–3.0 m out | 0 | test only: unfamiliar start |
| `oor_arm` | 30 | arm start offsets up to ±0.5 rad | 0 | test only: unusual arm pose |

The ranges themselves are in the `TRAIN` and `OOR` dictionaries at the top of `pipeline.py` (`dist`, `angle`, `height`, `start_xy`, `start_yaw`, `arm_offset`, `arm_offset_prob`).

### Step 1: plan (`STEP = "plan"`)

**What it does:** decides every episode in advance: target, start pose, arm offsets, noise. Episode IDs are numbered across all pools, so each ID means one episode only.

**You get:** `data/<pool>/plan.csv` and a coverage plot `outputs/plan_<pool>.png`.

**Note:** it refuses to overwrite an existing `plan.csv`. Delete it first only if nothing has been collected.

### Step 2: collect (`STEP = "collect"`)

**What it does:** runs the expert on every planned episode that doesn't have a data file yet.

**You get:**
- `data/<pool>/ep#####.parquet`: one row per 1 ms step (robot state, expert's commands, noise)
- `data/<pool>/summary.csv`: one row per episode

Key `summary.csv` columns:

| Column | Meaning |
|---|---|
| `stop_reason` | `reached`, `unreachable` (no 2 mm progress for 3 s), `step_cap`, or `crash` |
| `final_ee_error`, `best_ee_error` | final and best distance from hand to target (m) |
| `nav_done_time` | when the base finished parking (s); 25.0 = it timed out |
| `sim_duration` | episode length (s); always at least 8 s by design |
| `applied_offset0–6` | arm start offsets actually applied, after clipping |
| `frac_steps_pinned` | share of steps with a joint near its limit |

**Notes:**
- **Resumable:** after a crash or Ctrl+C, run the same command again; finished episodes are skipped.
- Run it outside VS Code, laptop plugged in, and don't open the live `summary.csv` in Excel (it locks the file).

### Step 3: qa (`STEP = "qa"`)

**What it does:** checks each pool and prints a report.

How to read the report:

| Line | Meaning | Good looks like |
|---|---|---|
| `planned X, finished Y, files Z` | episodes planned / with a summary row / with a data file | all three equal |
| stop-reason shares | how each episode ended | mostly `reached` |
| `by height` / `by distance` | stop reasons per height or distance band | no band much worse than the others |
| `MB/episode`, `s/episode` | file size and run time | for planning disk space and time |
| `RED FLAG: ...` | crashes, NaN values, clock errors, finger drift, success under 80% | none (except `dart`, where low success is expected) |
| `QA PASSED` | no red flags | |

**You get:** an outcome plot `outputs/outcomes_<pool>.png` (targets coloured by how the episode ended).

### Step 4: dataset (`STEP = "dataset"`)

**What it does:** turns each episode's raw log into model inputs (`X`) and labels (`y`).
- Base dataset: rows while the base is driving. Arm dataset: rows after it has parked.
- Failed episodes are cut at their closest approach; every `STRIDE`-th row is kept.

**Run it 3 times**, changing the settings each time:

| Run | `MODEL` | `RESIDUAL` | Creates |
|---|---|---|---|
| 1 | `"arm"` | `False` | `dataset_arm.npz`: labels = full arm torques |
| 2 | `"arm"` | `True` | `dataset_arm_residual.npz`: labels = arm torques minus gravity |
| 3 | `"base"` | `False` | `dataset_base.npz`: labels = right and left wheel commands |

Model inputs (built by `make_inputs`, in the robot's own frame):

| Model | Inputs |
|---|---|
| Base | target position relative to the robot, target height, distance and bearing to the target, forward / sideways speed, turning rate |
| Arm | target relative to the robot, 7 joint angles, 7 joint speeds, distance to the limit for joints 1, 3, 5, target minus shoulder (x, y, z), target minus hand (x, y, z) |

Each `.npz` holds `X`, `y`, `episode_id`, `phase`, `feature_names`, `label_names`.

---

## Step 5: check the datasets (optional)

**Run:** `pixi run python check_datasets.py` (no settings)

**What it checks:**
- **Joint wrap:** the arm joints without limits never turn a full circle in the data, so their raw angles are safe inputs.
- **Bearing:** when the target is behind the robot, the expert turns towards the target's side, so the raw bearing angle is the right input.

Each check prints a one-line verdict.

---

## Step 6: split the episodes

**Run:** `pixi run python splits.py`

**What it does:** splits the 420 `main` episodes into train (294), val (63) and test (63). The split is by **episode**, never by row, so near-identical rows can't end up on both sides.

| Setting (function arguments) | Meaning |
|---|---|
| `pool` | which pool to split (`"main"`) |
| `fractions` | train / val / test shares (`0.70, 0.15, 0.15`) |
| `seed` | fixed random seed (`0`) so anyone gets the same split |
| `overwrite` | `False` = refuses to replace an existing split, so the test set never changes by accident |

**You get:** `splits/train_ids.txt`, `val_ids.txt`, `test_ids.txt` and a table showing the splits have similar distances and heights. **Commit these files.**

---

## Step 7: train the models

**Run:** `pixi run python training.py`

**What it does:** for each dataset (`arm`, `arm_residual`, `base`) it trains:
- **Dummy:** always predicts the training average. This is the floor every real model must beat.
- **Ridge:** linear regression with a penalty on large weights; the penalty strength is chosen on the val split.

It saves each model and scores it on the **val** split. The test split is not used here.

| Setting | Meaning |
|---|---|
| `ALPHAS` | Ridge penalty strengths to try (higher = smaller weights, less overfitting) |
| `TORQUE_LIMIT` | torques are clipped to ±30 N·m when scored, as on the robot |


**Run:** `pixi run python mlp.py`

**What it does:** trains Multi-Layer Perceptron (MLP) neural network policies for both the base and arm models using the processed datasets. 
- **Base MLP:** Learns wheel commands from relative target position, distance, bearing, and robot speeds.
- **Arm MLP:** Learns joint torques (or residual torques) from relative target position, joint angles, joint speeds, and joint limit margins.

It trains using mini-batch gradient descent, tracks validation loss, and saves the best model weights.

| Setting | Meaning |
|---|---|
| `HIDDEN_SIZES` | Architecture of the hidden layers (e.g., `[128, 128]`) |
| `LEARNING_RATE` | Step size for the optimizer (e.g., `1e-3`) |
| `BATCH_SIZE` | Number of samples per training batch |
| `EPOCHS` | Total number of passes over the training dataset |
| `DROPOUT` | Regularization dropout rate (if applicable) |

How to read `outputs/results_offline.csv`:

| Column | Meaning |
|---|---|
| `val_nmse` | error divided by the error of always guessing the average: 1.0 = no better than the Dummy, 0 = perfect |
| `val_r2` | share of the variation explained: 1 = perfect, 0 = Dummy level |
| `val_rmse_<output>` | typical error per output in real units (N·m or rad/s) |
| `val_full_*` | arm only: scored on the full torque actually sent (gravity added back for the residual model) |
| `val_nmse_align/drive/orient` | base only: error in each phase of the expert's driving |
| `alpha` | the chosen Ridge penalty |
| `n_params`, `fit_s`, `infer_us` | model size, training time (s), time for one prediction (µs; the robot needs one every 1,000 µs) |

**You get:** one file per model, `models/<model>_<dataset>.joblib`, holding the model plus everything needed to use it (scaling numbers, feature names, label names).

---

## Step 8: closed-loop test

**Run:** `pixi run python closed_loop.py`

**What it does:** a trained model controls the robot on the same episodes the expert ran, and the results are compared. The expert still runs in the background, so its commands are logged next to the model's.

### Settings (top of `closed_loop.py`)

| Setting | Meaning |
|---|---|
| `NAME` | label for this run's result files (use a new one per experiment) |
| `ARM_MODEL` | path to a saved arm model, or `None` = the expert controls the arm |
| `BASE_MODEL` | path to a saved base model, or `None` = the expert drives the base |
| `SPLIT` | `"val"` while developing; `"test"` only for final results |
| `N_EPISODES` | `None` = every episode of the split; e.g. `3` for a quick try |
| `WORKERS` | parallel simulators |
| `GUI` | `True` = watch one episode in the 3D window |
| `SAVE_STEPS` | save every step of every episode (needed for step 9) |
| `POLICY_EVERY` | ask the model every N steps. **Keep at 1**: at 10 the arm became unstable |

### Typical sequence

1. `GUI = True`: watch one episode.
2. `GUI = False`, `N_EPISODES = 3`: check the table prints with no crashes.
3. `N_EPISODES = None`: all 63 val episodes.

**Run:** `pixi run python mlp_eval.py`

**What it does:** evaluates trained Multi-Layer Perceptron (MLP) models in closed-loop simulation on specified dataset splits, comparing their success rate and tracking performance against the expert controller.

| Setting | Meaning |
|---|---|
| `MODEL_PATH` | path to the saved MLP model weights (`.joblib` or `.pt`) |
| `SPLIT` | dataset split to evaluate on (`"val"` or `"test"`) |
| `GUI` | `True` = watch one episode in the 3D PyBullet window, `False` = headless execution |
| `N_EPISODES` | `None` = run every episode of the split, or a specified integer for a quick test |
| `WORKERS` | number of parallel simulation workers |

### How to read the printed table

| Column | Meaning |
|---|---|
| `success` | share of episodes ending with the hand within 1 cm of the target |
| `median_final_error_m` | typical final distance from hand to target (m) |
| `mean_time_s_when_reached` | average episode length when it succeeded |
| `crashes` | episodes that raised an error (the message is printed below the table) |

Rows: `expert` (from the collected data) and your model, on exactly the same episodes.

---

## Step 9: diagnose a run (optional)

**Run:** `pixi run python test.py` (set the model path and the `outputs/closed_loop/<NAME>/` folder inside it)

**What it does:** for one closed-loop episode, it plots the hand-to-target distance and, for each joint, the torque the model sent against the torque the expert would have sent.

**You get:** `outputs/diagnose_*.png`. Use a new file name per run, or the plots overwrite each other.

---

## GRU policy (`gru.py`)

**Final model: `models/gru_e80.joblib`** (Sam)

A recurrent policy built on top of the pipeline above, the same way the MLP files are (no pipeline file is edited). The base GRU drives until the base is parked, then the arm GRU takes over and reaches. The memory is there because the expert isn't a pure function of the current state: it switches between align / drive / orient while driving, and the arm controller builds up its own joint reference `q_des` over time. A single-step model can't see either of those.

- **Inputs:** base = 10 features (target and parking pose relative to the robot, heading errors). Arm = 39 features (target, joint angles as sin/cos and raw, joint speeds, hand-to-target error).
- **Outputs:** base = 2 wheel speeds (clipped to ±3.5 rad/s). Arm = 7 torques minus gravity (gravity added back at run time, clipped to ±30 N·m).
- **Network (each):** 128 embedding → 1 GRU layer of 128 → 2×128 head that sees the memory and the current input. The memory updates every 10 ms; the head runs every 1 ms. 305k parameters in total (base 150k, arm 155k), 2.4 MB file.
- **Training:** whole episodes as sequences (every 10th step), batches of 8 episodes, MSE on standardised labels with arm steps within 5 cm of the target weighted 3×, AdamW 1e-3 with cosine decay, weight decay 1e-5, gradient clip 1.0. Trains for a fixed number of epochs and keeps the epoch with the lowest val loss.
- **Running it:** plain numpy, about 100 µs per step against the 1 ms control period. Torque is only needed for training.

### How to run

Set the settings block at the **bottom of `gru.py`**, then `pixi run python gru.py`.

| `STEP` | What it does | Output |
|---|---|---|
| `"pools"` | Plans and collects the two pools `pipeline.py` doesn't define: `dart2` (500 episodes, DART noise 0.5) and `sel` (137 clean episodes only used to choose between models) | `data/dart2/`, `data/sel/` |
| `"train"` | Trains the base and arm GRUs on `main` train + the `EXTRA` pools | `models/<NAME>.joblib` |
| `"closed_loop"` | The model (or the expert, with `NAME = "expert"`) drives on `SPLIT` of `main`, or on another `POOL`, optionally with noise / low friction | `outputs/closed_loop_<NAME>_<split or pool><TAG>.csv` |
| `"report"` | Size, inference time, offline test NMSE and closed-loop numbers, plus the loss curves | `outputs/gru_report.csv`, `outputs/gru_loss_curves_<NAME>.png` |
| `"ablation"` | The development table in the report (offline NMSE + val / sel successes for each version) | `outputs/gru_ablation.csv` |
| `"dagger"`, `"ensemble"` | The two experiments that didn't make it into the final model (see below) | `data/gru_dagger1/`, `models/<NAME>.joblib` |

| Setting | Meaning |
|---|---|
| `NAME` | model name in `models/`; for `closed_loop` it can also be `"expert"` |
| `EXTRA` | extra training pools, e.g. `["dart", "dart2"]` |
| `EPOCHS` | epochs (the best val epoch is kept, there is no early stopping) |
| `BASE_FROM` | copy the base GRU from this model and only train the arm (`None` = train both) |
| `NEAR_WEIGHT`, `NEAR_DIST` | arm steps closer than `NEAR_DIST` m to the target count `NEAR_WEIGHT` times |
| `SEED` | training seed (starting weights and batch order only) |
| `SPLIT`, `POOL` | which episodes `closed_loop` uses: `SPLIT` of `main`, or `POOL = "sel"` / `"oor_far"` / `"oor_high"` / `"oor_start"` / `"oor_arm"` |
| `SENSOR_NOISE`, `ACT_NOISE`, `FRICTION` | robustness conditions, same definitions as `mlp_eval.py` |
| `TAG` | added to the output file name for robustness runs so the clean result isn't overwritten |
| `WORKERS`, `GUI` | parallel simulators; `GUI = True` shows one episode in the window |

**Before you start:** `pipeline.py` needs to have collected `main`, `dart` and the four `oor_*` pools, and `splits.py` needs to have made the splits (steps 1–6 above). Then run `STEP = "pools"` once.

**To just watch it:** `pixi run python run_random.py` puts the target somewhere random and lets `gru_e80` drive in the pybullet window (settings at the top of the file, `POLICY = "expert"` to watch the expert).

### Rebuilding every model in the report

Run `"train"` with these settings, in this order (anything not listed is the default: `NEAR_WEIGHT, NEAR_DIST = 3.0, 0.05`, `SEED = 0`).

| Model | `EXTRA` | `EPOCHS` | `BASE_FROM` | Other | Notes |
|---|---|---|---|---|---|
| `gru` | `["dart"]` | 40 | `None` | | first version, main + DART 150 |
| `gru_dart2` | `["dart", "dart2"]` | 40 | `None` | | + DART 500 (650 DART episodes in total) |
| `gru_nw10` | `["dart", "dart2"]` | 40 | `"gru_dart2"` | `NEAR_WEIGHT, NEAR_DIST = 10.0, 0.03` | 10× weight on the last 3 cm (discarded) |
| **`gru_e80`** | `["dart", "dart2"]` | **80** | `"gru_dart2"` | | **final model** |
| `gru_e80_s1`, `gru_e80_s2` | `["dart", "dart2"]` | 80 | `"gru_dart2"` | `SEED = 1` / `2` | only needed for the ensemble |
| `gru_ens` | | | | `STEP = "ensemble"`, `MEMBERS` as in the file | mean of the three 80-epoch arms (discarded) |
| `gru_dagger1` | `["dart", "dart2", "gru_dagger1"]` | 40 | `"gru_dart2"` | first run `STEP = "dagger"` with `NAME = "gru_dart2"` | DAgger round 1 (discarded) |

All of these except the seed models are in `models/`, so you can skip training and go straight to `"closed_loop"`.

### Rebuilding the results

| Result (report) | Settings for `STEP = "closed_loop"` | File |
|---|---|---|
| Development table, val / sel | `NAME` = each model, `SPLIT = "val"`, then `POOL = "sel"` | `closed_loop_<model>_val.csv`, `_sel.csv` |
| Development table, NMSE column | `STEP = "ablation"` | `gru_ablation.csv` |
| Final test result | `NAME = "gru_e80"`, `SPLIT = "test"` | `closed_loop_gru_e80_test.csv` |
| Out-of-range | `POOL = "oor_far"` / `"oor_high"` / `"oor_start"` / `"oor_arm"` | `closed_loop_gru_e80_oor_*.csv` |
| Robustness, GRU | `SPLIT = "val"`, then one at a time: `ACT_NOISE = 0.25, TAG = "_act025"`; `ACT_NOISE = 0.5, TAG = "_act05"`; `FRICTION = 0.4, TAG = "_fr04"`; `SENSOR_NOISE = 0.005 / 0.01 / 0.02, TAG = "_sn005" / "_sn01" / "_sn02"` | `closed_loop_gru_e80_val_<tag>.csv` |
| Robustness, expert | same, with `NAME = "expert"` (the expert reads the true state, so only actuator noise and friction apply) | `closed_loop_expert_val_<tag>.csv` |
| Size / speed / loss curves | `STEP = "report"` | `gru_report.csv`, `gru_loss_curves_gru_e80.png` |

Everything is deterministic: the same model on the same episode gives the same result every time.

### Results

**Final model** (`gru_e80`, test split used once at the end):

| | GRU | Expert |
|---|---|---|
| Test success | 60/63 (95.2%) | 63/63 |
| Test final error | 0.92 ± 0.41 cm | 0.86 ± 0.29 cm |
| Val / sel (used to choose it) | 62/63, 133/137 | |

**How it was chosen** (`outputs/gru_ablation.csv`): every version was compared on closed-loop success over 200 episodes (val 63 + `sel` 137), since 63 alone is too few to separate models that all score 90–95%. Lower offline NMSE didn't always mean better driving.

| Version | Arm NMSE (test, offline) | Val (/63) | Sel (/137) |
|---|---|---|---|
| `gru` (main + DART 150) | 0.050 | 51 | – |
| `gru_dart2` (+ DART 500) | 0.034 | 58 | 127 |
| `gru_dagger1` (discarded) | 0.054 | 51 | 98 |
| `gru_nw10` (discarded) | 0.047 | 58 | 130 |
| **`gru_e80` (final)** | **0.027** | **62** | **133** |
| `gru_ens` (discarded) | 0.024 | 61 | 132 |

Retraining with a different seed moved the score by about ±5 of 200, so the ensemble and the 10× weighting weren't real improvements.

**Generalisation** (30 episodes each, never seen in training):

| Pool | GRU |
|---|---|
| `oor_far` (targets 2.5–3.5 m away) | 30/30 |
| `oor_high` (targets 1.10–1.30 m high) | 12/30 |
| `oor_start` (start 1.5–3.0 m out) | 29/30 |
| `oor_arm` (arm offsets up to 0.5 rad) | 28/30 |

**Robustness** (val, 63 episodes, expert rerun under the same conditions):

| Condition | GRU | Expert |
|---|---|---|
| none | 62 | 63 |
| actuator noise σ = 0.25 / 0.5 | 59 / 53 | 57 / 41 |
| floor friction μ = 0.4 | 62 | 63 |
| sensor noise σ = 0.005 / 0.01 / 0.02 rad | 62 / 62 / 63 | – |

**Against the MLP** on the same test episodes: `outputs/gru_vs_mlp.csv` (the `test + ...` rows are earlier robustness runs on the test split, kept for that comparison).

**Notes:**
- **DAgger made it worse** (98/137 on `sel` against 127 without it). The expert's torque depends on its own `q_des`, which winds up to its 0.15 rad limit while the GRU is driving, so the labels it records are one long saturated push rather than what it would normally do from that state.
- **The 6 slow test episodes** are ones where the base hit the 25 s navigation timeout and the arm then reached from wherever the base stopped. Mean time to reach is 11.8 s (10.0 s without those 6).
- **`dart2` settings were rebuilt** from `outputs/plan_dart2.png`: noise 0.5, IDs 710–1209. `sel` uses IDs 1210–1346.

### What's reused and what's new

We edited: robotnik_arm_task.py to facilitate policy input and training as well as closed loop testing. Noah wrote: closed_loop.py, robotnik_arm_task.py, pipeline.py, splits.py, test.py and training.py as the base files. Used primarily a mix of tutorial from sci-kit learn, pandas, numpy and previous labs with minimal ai usage to create training pipeline and base for data generation. 

Finn wrote mlp.py and mlp_eval.py, which were based off of Noah's pipeline files. mlp.py can be used instead of training.py to train the mlp models and mlp_eval.py can be used instead of closed loop to test the mlp policy with the error policy. Cluade Code was used as an assit to help with this part of the project.

Sam wrote gru.py and ai tools was used to assist.

---

## Output files

```
data/<pool>/plan.csv                    planned episodes
data/<pool>/ep#####.parquet             expert episode, one row per 1 ms step
data/<pool>/summary.csv                 one row per expert episode
data/<pool>/dataset_*.npz               model inputs (X) and labels (y)
splits/*_ids.txt                        episode IDs of each split
models/<model>_<dataset>.joblib         trained models
outputs/results_offline.csv             offline scores (val split)
outputs/closed_loop_<NAME>.csv          closed-loop result per episode
outputs/closed_loop/<NAME>/*.parquet    closed-loop step logs (incl. the expert's commands)
outputs/*.png                           coverage, outcome and diagnosis plots
```

Data and trained models are not in the repo; regenerate them with the steps above. [might be]


