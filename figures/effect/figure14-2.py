import os
import sys
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import palettable

stop_list = ['', '\n']

def read_float(filename, split = " "):
    assert isinstance(filename, str)
    assert isinstance(split, str)
    return_list = []
    with open(filename, 'r') as f:
        tmpstr = f.readline()
        while(tmpstr):
            tmplist = re.split(split, tmpstr)
            reslist = []
            for i in stop_list:
                try:
                    tmplist.remove(i)
                except:
                    pass
            for i in tmplist:
                tmp = float(i)
                assert tmp > 0
                reslist.append(tmp)
            return_list.append(reslist)
            tmpstr = f.readline()
    return return_list

font_size = 30
plt.rc('font',**{'size': font_size, 'family': 'Arial'})
plt.rc('pdf',fonttype = 42)
fig_size = (9, 6)
fig, axes = plt.subplots(figsize=fig_size)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=None)

file_name = ['../../Auncel/eval/Effective_time_sift10M.log', '../../Auncel/eval/Effective_time_deep10M.log', 
             '../../Auncel/eval/Effective_time_gist.log', '../../Auncel/eval/Effective_time_text.log']
res = read_float(file_name[1])
data = {i: [] for i in range(5, 55, 5)}
for i in res:
    data[int(i[0])].append(i[1])
# for i in data:
#     print(len(data[i]), sum(data[i])/len(data[i]), min(data[i]), max(data[i]))
x = [i for i in data]
y = [sum(data[i])/len(data[i]) for i in data]
# y_low = [sum(data[i])/len(data[i]) - min(data[i]) for i in data]
# y_high = [max(data[i]) - sum(data[i])/len(data[i]) for i in data]
# y_err = [y_low, y_high]

y_best = [min(data[i]) for i in data]
y_worst = [max(data[i]) for i in data]

d1 = np.abs(np.array(x) - np.array(y_best))
d2 = np.abs(np.array(x) - np.array(y_worst))
# print(d1)
# print(d2)
# print(np.max(d1))
# print(np.max(d2))

# Plot bars
# top
l2 = axes.plot(x, y_worst, label='Maximum Time', marker='^', color='r', linewidth=3.6, markersize=11, zorder=4)
l1 = axes.plot(x, y_best, label='Minimum Time', marker='o', color='royalblue', linewidth=3.6, markersize=11, zorder=3)

axes.set_xlim(left=5, right=50)
axes.set_ylim(bottom=0, top=55)

# Set top and right axes to none
# ax1.spines['bottom'].set_visible(False)
#     axes[j].spines['top'].set_visible(False)
#     axes[j].spines['right'].set_visible(False)
axes.set_xlabel('Requested Response Time (ms)')
axes.set_ylabel('Actual Response Time (ms)')
x_ticks = [i for i in range(5, 55, 5)]
axes.set_xticks(x_ticks)
y_ticks = [i for i in range(0, 60, 10)]
axes.set_yticks(y_ticks)
axes.grid(axis='y', linestyle='--')
# ax1.get_yaxis().set_tick_params(pad=12)
axes.legend(frameon=False, ncol=1, loc='upper center', bbox_to_anchor=(0.3, 1.0), prop={'size': font_size})

# Save the figure
file_path = '14-2.pdf'
plt.savefig(file_path, bbox_inches='tight', transparent=True, backend='pgf')