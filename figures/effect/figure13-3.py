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
                # assert tmp > 0
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

file_name = ['../../Auncel/eval/Effective_error_sift10M.log', '../../Auncel/eval/Effective_error_deep10M.log', 
             '../../Auncel/eval/Effective_error_gist.log', '../../Auncel/eval/Effective_error_text.log']
res = read_float(file_name[2])
data = {i: [] for i in range(10, 80, 10)}
for i in res:
    data[int(100 - i[0]*100)].append(int(100 - i[1]*100))
# for i in data:
#     print(len(data[i]), sum(data[i])/len(data[i]), min(data[i]), max(data[i]))
x = [i for i in data]
for i in data:
    data[i].sort()
y_tail95 = [data[i][int(0.95*len(data[i]))] for i in data]
y_worst = [max(data[i]) for i in data]

# Plot bars
# top
l1 = axes.plot(x, x, label='Ideal', marker=None, color='k', lw=2.4, zorder=4)
l3 = axes.plot(x, y_worst, label='Maximum Error', marker='^', color='r', lw=3.6, zorder=3, markersize=11)
l2 = axes.plot(x, y_tail95, label='95%-tile Error', marker='o', color='royalblue', lw=3.6, zorder=3, markersize=11)

axes.set_xlim(left=10, right=70)
axes.set_ylim(bottom=0, top=100)

axes.set_xlabel('Requested Error Bound (%)')
axes.set_ylabel('Actual Error (%)')
x_ticks = [i for i in data]
axes.set_xticks(x_ticks)
axes.grid(axis='y', linestyle='--')
# ax1.get_yaxis().set_tick_params(pad=12)
axes.legend(frameon=False, ncol=1, loc='upper center', bbox_to_anchor=(0.3, 1.0), prop={'size': font_size})

# Save the figure
file_path = '13-3.pdf'
plt.savefig(file_path, bbox_inches='tight', transparent=True, backend='pgf')