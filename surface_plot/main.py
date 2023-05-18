import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.axes3d import Axes3D

df = pd.read_csv("fes.csv", sep="\t")

x, y = df["CV1"].round(2).unique(), df["CV2"].unique()
X, Y = np.meshgrid(x, y)

z = df["free energy (kJ/mol)"]
Z = np.zeros_like(X)



for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i][j] = z[(i*X.shape[0])+j]
        # print("done for",i,j)

print("===== X =====\n", X, f"\n===== {X.shape} =====\n")
print("===== Y =====\n", Y, f"\n===== {Y.shape} =====\n")
print("===== Z =====\n", Z, f"\n===== {Z.shape} =====\n")

fig = plt.figure(figsize=(16,6.5))

ax = fig.add_gridspec(1,5)

ax_x1 = fig.add_subplot(ax[0, :2])
ax_x1.contour(X,Y,Z,levels=20)

ax_x1.set_xlabel("CV1")
ax_x1.set_ylabel("CV2")

ax_x2 = fig.add_subplot(ax[0, 2:], projection="3d")
ax_x2.contour(X,Y,Z, zdir="z", offset=-1, levels=20)
plot_sur = ax_x2.plot_surface(X,Y,Z, cmap="viridis")

ax_x2.set_xlabel("CV1")
ax_x2.set_ylabel("CV2")
ax_x2.set_zlabel("free energy (kJ/mol)")

fig.colorbar(plot_sur, shrink=0.5)

# plt.show()

plt.savefig("plot.pdf")