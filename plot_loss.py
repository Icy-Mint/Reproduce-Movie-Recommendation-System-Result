import pickle, matplotlib.pyplot as plt, argparse, numpy as np, pathlib

p = argparse.ArgumentParser(); p.add_argument("--fold", type=int, required=True)
args = p.parse_args()

hist = pickle.load(open(f"data100k/f{args.fold}/loss.pkl","rb"))
plt.plot(hist["loss"], label="train")
plt.plot(hist["val_loss"], label="val")
plt.ylabel("MSE"); plt.xlabel("epoch"); plt.legend()
pathlib.Path("Figs").mkdir(exist_ok=True)
plt.savefig(f"Figs/loss_curve_f{args.fold}.png", dpi=300)
print("✓ loss curve ready")
