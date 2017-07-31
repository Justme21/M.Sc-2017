#!/usr/bin/python

import matplotlib.pyplot as plt
import sys

plot_file = None
row_cont = None
for entry in sys.argv[1:]:
    plot_file = open(entry,"r")
    row_cont = []
    for row in plot_file:
        row = row.split()
        row_cont += [float(x.strip(' ,')) for x in row]
    plot_file.close()
    plt.plot(row_cont)

plt.show()
