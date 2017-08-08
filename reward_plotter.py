#!/usr/bin/python

import matplotlib.pyplot as plt
import sys

plot_file = None
row_cont = None
do_plot = True
for entry in sys.argv[1:]:
    plot_file = open(entry,"r")
    row_cont = []
    for row in plot_file:
        if "RESULTS" in row: do_plot = False
        if "REWARDS:" in row:
            do_plot = True
            row = row[9:]
        if do_plot:
            row = row.split()
            row_cont += [float(x.strip('[,]')) for x in row]
    plot_file.close()
    plt.plot(row_cont)

plt.show()
