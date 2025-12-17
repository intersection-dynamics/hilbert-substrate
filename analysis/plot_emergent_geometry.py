import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "emergent_geometry_results.npz"

data = np.load(DATA_FILE, allow_pickle=True)
t = data["t"]
mz = data["mz"]            # (N, T+1)
dist = data["distances"]   # (N,)
r = data["ball_radii"]
B = data["ball_volumes"]
r_front = data["r_front"]

# 1) Ball volumes
plt.figure()
plt.plot(r, B, marker="o")
plt.xlabel("graph radius r")
plt.ylabel("B(r)")
plt.title("Ball volumes in interaction graph")
plt.grid(True)

# 2) Front radius vs time
plt.figure()
plt.plot(t, r_front, marker=".")
plt.xlabel("time step")
plt.ylabel("r_front(t)")
plt.title("Effective front radius vs time")
plt.grid(True)

# 3) Heatmap of <Z_j>(t)
plt.figure()
plt.imshow(
    mz,
    aspect="auto",
    origin="lower",
    extent=[t[0], t[-1], 0, mz.shape[0] - 1],
)
plt.xlabel("time step")
plt.ylabel("node index j")
plt.colorbar(label="<Z_j>")
plt.title("<Z_j>(t) on random interaction graph")

plt.show()
