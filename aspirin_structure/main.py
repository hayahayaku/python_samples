import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import distance_matrix

asp_NaN = pd.read_fwf("https://raw.githubusercontent.com/baoilleach/qmspeedtest/master/INPUTFILES/aspirin.xyz", skiprows=[0,1], header=None)

asp_list = []

for x in range(asp_NaN.shape[0]):
    ret = list(asp_NaN.iloc[x].dropna().values)

    # atom size
    if ret[0] == "H": ret.append(12.0**2)
    elif ret[0] == "C": ret.append(17.0**2)
    elif ret[0] == "O": ret.append(15.2**2)

    # atom color
    if ret[0] == "H": ret.append('darkgrey')
    elif ret[0] == "C": ret.append('black')
    elif ret[0] == "O": ret.append('red')

    asp_list.append(ret)

asp = pd.DataFrame(asp_list)


fig = plt.figure()
ax1 = fig.add_subplot(projection="3d")

ax1.scatter(asp[1],asp[2],asp[3], s=asp[4], c=asp[5])

# print(list(asp[:,1:4]))
dm = distance_matrix(asp.iloc[:,1:4], asp.iloc[:,1:4])

where = np.argwhere(dm < 1.6)

# print(where)

conns = []

for coor in where:
    if coor[0] >= coor[1]:
        pass
    else:
        x = list(asp.iloc[coor[0],1:4])
        y = list(asp.iloc[coor[1],1:4])
        conns.append([x,y])

conn_lines = Line3DCollection(conns, edgecolor="grey", linestyle="solid", linewidth=3)

ax1.add_collection3d(conn_lines)
ax1.view_init(45,45,45)

ax1.set_title("Aspirin 3D structure")

# plt.show()

plt.savefig("plot.pdf")