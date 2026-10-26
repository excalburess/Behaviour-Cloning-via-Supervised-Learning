import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from policy import make_mlp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#Load dataset in
d = np.load("data/dataset.npz")
X_train, Y_train = d["X_train"], d["Y_train"]
X_val, Y_val = d["X_val"], d["Y_val"]
X_test, Y_test = d["X_test"], d["Y_test"]
x_mean, x_std, y_mean, y_std = d["x_mean"], d["x_std"], d["y_mean"], d["y_std"]

def norm_x(X):
    return ((X-x_mean)/x_std)

def norm_y(Y):
    return ((Y-y_mean)/y_std)

def denorm_y(Y):
    return ((Y*y_std)+y_mean)

#------------
# Linear Regression model
#------------

A = np.column_stack([norm_x(X_train), np.ones(len(X_train))])
W, *_= np.linalg.lstsq(A, norm_y(Y_train), rcond=None)

A_val = np.column_stack([norm_x(X_val), np.ones(len(X_val))])

Y_pred = A_val @ W #calc predicted value
Y_pred = denorm_y(Y_pred)#denormalise the output
err = Y_pred - Y_val
rmse_linear = np.sqrt(np.mean(err**2, axis=0))#root mean square error

OUT_NAMES = [f"wheel_{i}" for i in range(1, 5)] + [f"torque_{i}" for i in range(1, 8)]
y_val_std = Y_val.std(axis=0)
for name, r, s in zip(OUT_NAMES, rmse_linear, y_val_std):
    print(f"{name:10s} RMSE {r:7.3f}   std {s:7.3f}   ratio {r/s:5.2f}")

#-----------
#Py torch neural netweork
#------------------

Xtr_t = torch.tensor(norm_x(X_train), dtype=torch.float32)
Ytr_t = torch.tensor(norm_y(Y_train), dtype=torch.float32)
Xval_t = torch.tensor(norm_x(X_val), dtype=torch.float32)
Yval_t = torch.tensor(norm_y(Y_val), dtype=torch.float32)
train_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=1024, shuffle=True)



model = make_mlp()
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
best_val = float("inf")
train_losses, val_losses = [], []
n_epochs = 30
t0 = time.time()
#Training loop
for epoch in range(n_epochs):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        running += loss.item()
    train_loss = running / len(train_loader)

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(Xval_t), Yval_t).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "model_best.pt")
    print(epoch + 1, train_loss, val_loss)
train_time = time.time() - t0
#Loss curves
plt.plot(range(1, n_epochs + 1), train_losses, label="train")
plt.plot(range(1, n_epochs + 1), val_losses, label="val")
plt.xlabel("epoch"); plt.ylabel("MSE (normalised)"); plt.yscale("log"); plt.legend()
plt.savefig("loss_curve.png", dpi=150)
 
# ---------- 5. Evaluate the best model in real units ----------
model.load_state_dict(torch.load("model_best.pt"))
model.eval()
with torch.no_grad():
    Y_pred = denorm_y(model(Xval_t).numpy())
rmse = np.sqrt(np.mean((Y_pred - Y_val) ** 2, axis=0))
for name, r, s in zip(OUT_NAMES, rmse, Y_val.std(axis=0)):
    print(f"{name:9s} rmse {r:.3f}  std {s:.3f}  ratio {r / s:.3f}")
print(f"mean rmse {rmse.mean():.3f}")
 
# ---------- 6. Cost numbers for the results table ----------
n_params = sum(p.numel() for p in model.parameters())
x1 = Xval_t[:1]
with torch.no_grad():
    t1 = time.perf_counter()
    for _ in range(1000):
        model(x1)
    infer_ms = (time.perf_counter() - t1)  # 1000 calls, so seconds == ms per call
print(f"params {n_params}  train time {train_time:.1f}s  inference {infer_ms:.3f} ms/sample")

ep_train = d["ep_train"]
run_ids = np.unique(ep_train)
rng = np.random.default_rng(0)
for n_runs in [25, 50, 100, len(run_ids)]:
    keep = rng.choice(run_ids, n_runs, replace=False)
    mask = np.isin(ep_train, keep)
    X_sub, Y_sub = X_train[mask], Y_train[mask]
    # ...train on X_sub, Y_sub exactly as before, record best_val...