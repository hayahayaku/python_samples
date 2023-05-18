import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {}

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot()

plt.grid(visible=True)

ax1.set_yscale("log")

# load data table
for each in ['BY','NW','SN','TH']:
    data[each] = pd.read_csv(f"https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/master/data/de-states/de-state-{each}.tsv", sep="\t")
    plt.plot(data[each]["Date"], data[each]["Cases_Last_Week_Per_Million"], label=each)

# find global maximum
max = { "state": None, "cases": 0, "date": None }
for each in data:
    max_state = data[each]["Cases_Last_Week_Per_Million"].max()
    max_index = data[each]["Cases_Last_Week_Per_Million"].argmax()
    if max_state > max["cases"]:
        max["state"] = each
        max["cases"] = max_state
        max["date"] = data[each]["Date"][max_index]

# label global maximum, modified from
# https://stackoverflow.com/questions/43374920/how-to-automatically-annotate-maximum-value-in-pyplot
xmax = max["date"]
ymax = max["cases"]
text= "Maximum n={} in {} @{}".format(ymax, max["state"], xmax)
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
arrowprops=dict(arrowstyle="simple")
kw = dict(xycoords='data',textcoords="axes fraction",
            arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
ax1.annotate(text, xy=(xmax, ymax), xytext=(0.95,1), **kw)

date_count = data['BY'].__len__() # 1061

ax1.set_title("7-day incidence/Mil of Covid cases")

ax1.legend(loc=2, frameon=False)
ax1.set_xlabel("Date")
ax1.set_ylabel("n/(week * million)")
ax1.set_xticks(data["BY"]["Date"][range(0,date_count,125)])

# inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
inset_ax = inset_axes(ax1, "100%", "100%", loc="lower right",
    # relative to the parent plot, from bottom left corner
    # x-origin, y-origin, x-size, y-size
    bbox_to_anchor=(0.6,0.05,0.35,0.25),
    bbox_transform=ax1.transAxes)

inset_ax.set_yscale("log")
data_de = pd.read_csv(f"https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/master/data/de-states/de-state-DE-total.tsv", sep="\t")
plt.plot(data_de["Date"], data_de["Cases_Last_Week_Per_Million"])

inset_ax.set_title("Incidence in whole Germany")
inset_ax.set_xticks(data["BY"]["Date"][range(0,date_count,300)])
plt.grid(visible=True)
# ax1.grid()
# inset_ax.legend()
# inset_ax.set_xlabel("Date")
# inset_ax.set_ylabel("n/(week * million)")

plt.savefig("plot.pdf", dpi=20)