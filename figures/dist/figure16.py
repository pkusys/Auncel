import os
import sys
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import palettable

font_size = 30
plt.rc('font',**{'size': font_size, 'family': 'Arial'})
plt.rc('pdf',fonttype = 42)
fig_size = (9, 6)
fig, axes = plt.subplots(figsize=fig_size)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=None)

# x = [4, 8, 16, 32, 64, 128]
x = [2, 3, 4, 5, 6, 7]
cal = [330, 115, 61.4, 49.7, 32.2, 24.2]
cal_no = [267, 113.4, 50.5, 37.5, 25.1, 19.2]

# Plot bars
# top
# l3 = axes.plot(x, cal_no, label='Without Calibration', marker='^', color='k', lw=3.6, zorder=2, markersize=11, linestyle='dashed')
l2 = axes.plot(x, cal, label='With Calibration', marker='o', color='r', lw=3.6, markersize=11, linestyle='solid',zorder=3)

# axes.set_xlim(left=10, right=70)
axes.set_ylim(bottom=0, top=350)

axes.set_ylabel('Average Latency (ms)')
axes.set_xlabel('No. of Workers')
# axes.set_xscale('log', base=2)
x_ticklabels = ['4', '8', '16', '32', '64', '128']
axes.set_xticklabels(x_ticklabels)
axes.set_xticks(x)
axes.grid(axis='y', linestyle='--')
y_ticks = [i for i in range(0, 400, 50)]
axes.set_yticks(y_ticks)


# Save the figure
file_path = 'figure16.pdf'
plt.savefig(file_path, bbox_inches='tight', transparent=True)