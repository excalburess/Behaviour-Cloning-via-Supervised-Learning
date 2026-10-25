"""Turn raw expert runs (data/train, data/test) into a training-ready dataset (data/dataset.npz)."""
import glob
import numpy as np
from features import make_features, FEATURE_NAMES

K = 10            # keep every k-th time step
VAL_FRAC = 0.2    # fraction of training runs held out for validation
SPLIT_SEED = 0


def load_runs(folder):
    runs = []
    dropped_runs = []
    for path in sorted(glob.glob(f"{folder}/run_*.npz")):
        d = dict(np.load(path))
        if str(d["stop_message"]).startswith("Target reached"):
            has_nan = any(np.isnan(v).any() for v in d.values() if v.dtype.kind == "f")
            if has_nan:
                dropped_runs.append((path, "has_nan"))
            else:
                runs.append(d)
        else:
            dropped_runs.append((path, str(d["stop_message"])))
    print(f"{folder}: kept {len(runs)}, dropped {len(dropped_runs)}")
    for p, why in dropped_runs:
        print("  dropped", p, "-", why)
    return runs


def make_labels(d):
    return np.concatenate([d["wheel_cmds"], d["torque_arm"]], axis=1)   # (N, 4 + 7)


def build_split(runs, k):
    X_list, Y_list, ep_list = [], [], []
    for i, d in enumerate(runs):
        X = make_features(d)[::k]
        Y = make_labels(d)[::k]
        X_list.append(X)
        Y_list.append(Y)
        ep_list.append(np.full(len(X), i))   # which run each row came from
    return (np.concatenate(X_list, axis=0),
            np.concatenate(Y_list, axis=0),
            np.concatenate(ep_list))


def main():
    # Step B: load, then split train/val by run
    runs = load_runs("data/train")
    test_runs = load_runs("data/test")
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(runs))
    n_val = max(1, int(VAL_FRAC * len(runs)))   # at least one validation run
    val_runs = [runs[i] for i in idx[:n_val]]
    train_runs = [runs[i] for i in idx[n_val:]]
    print(f"runs: train {len(train_runs)} | val {len(val_runs)} | test {len(test_runs)}")

    # Steps C-E: features, labels, downsampling
    X_train, Y_train, ep_train = build_split(train_runs, K)
    X_val, Y_val, ep_val = build_split(val_runs, K)
    X_test, Y_test, ep_test = build_split(test_runs, K)
    print("X_train", X_train.shape, "Y_train", Y_train.shape)
    print("X_val  ", X_val.shape, "Y_val  ", Y_val.shape)
    print("X_test ", X_test.shape, "Y_test ", Y_test.shape)

    # Step F: normalisation statistics from the TRAINING set only
    x_mean, x_std = X_train.mean(0), X_train.std(0) + 1e-8
    y_mean, y_std = Y_train.mean(0), Y_train.std(0) + 1e-8

    # Saved un-normalised; apply (X - x_mean) / x_std when training and in the simulator
    np.savez_compressed(
        "data/dataset.npz",
        X_train=X_train, Y_train=Y_train, ep_train=ep_train,
        X_val=X_val, Y_val=Y_val, ep_val=ep_val,
        X_test=X_test, Y_test=Y_test, ep_test=ep_test,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        feature_names=np.array(FEATURE_NAMES), k=K,
    )
    print("saved data/dataset.npz")


if __name__ == "__main__":
    main()
