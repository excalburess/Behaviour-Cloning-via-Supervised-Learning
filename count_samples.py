import numpy as np
for m in ("base", "arm"):
    print(m, len(np.load(f"data/main/dataset_{m}.npz")["X"]))