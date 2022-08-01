import re
import pickle
import numpy as np
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import palettable

def takesec(ins):
    return ins[1]

stop_list = ['', '\n']

def read(filename, split = " "):
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
                reslist.append(float(i))
            return_list.append(reslist)
            tmpstr = f.readline()
    res = [i[0] * 1000 for i in return_list]
    return np.array(res)

lat_slow_elp = []
lat_slow_laet = []
lat_slow_faiss = []

ave_lat_elp = []
ave_lat_laet = []
ave_lat_faiss = []

f11 = '../../Auncel/eval/Auncel_Latency_sift10M_100_10.log'
f12 = '../../LAET/benchs/learned_termination/LAET_Latency_sift10M_100_10.log'
f13 = '../../faiss/eval/Faiss_Latecy_sift10M_100_10.log'

f21 = '../../Auncel/eval/Auncel_Latency_sift10M_50_10.log'
f22 = '../../LAET/benchs/learned_termination/LAET_Latency_sift10M_50_10.log'
f23 = '../../faiss/eval/Faiss_Latecy_sift10M_50_10.log'

f31 = '../../Auncel/eval/Auncel_Latency_sift10M_10_10.log'
f32 = '../../LAET/benchs/learned_termination/LAET_Latency_sift10M_10_10.log'
f33 = '../../faiss/eval/Faiss_Latecy_sift10M_10_10.log'

logs = [[f31,f32,f33], [f21, f22, f23], [f11,f12,f13]]

# average
for i in logs:
    elp = read(i[0])
    laet = read(i[1])
    faiss = read(i[2])

    ave_lat_elp.append(elp.mean())
    ave_lat_laet.append(laet.mean())
    ave_lat_faiss.append(faiss.mean())

    lat_slow_elp.append(1.)

    cnt1 = 0.0
    cnt2 = 0.0
    for j in range(len(elp)):
        cnt1 += laet[j] / elp[j]
        cnt2 += faiss[j] / elp[j]
    cnt1 /= len(elp)
    cnt2 /= len(elp)

    lat_slow_laet.append(cnt1)
    lat_slow_faiss.append(cnt2)

# Set font and figure size
font_size = 30
plt.rc('font',**{'size': font_size, 'family': 'Arial'})
plt.rc('pdf',fonttype = 42)
fig_size = (12, 6)
fig, ax1 = plt.subplots(figsize=fig_size)
# fig.subplots_adjust(hspace=0.18)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=None)

# data
sys_name = 'Auncel'
datasets = ['Top-10', 'Top-50', 'Top-100']

# constants
num_datasets = 3
num_bars = 3

# indexes for x
width = 0.04
indexes = [[0.2*i+0.12+width*j for i in range(num_datasets)] for j in range(num_bars)]

##############################################
#  ax1
##############################################

# Plot bars
# top
b1 = ax1.bar(indexes[0], ave_lat_elp, width, label=sys_name, edgecolor='black',zorder=3)
b2 = ax1.bar(indexes[1], ave_lat_laet, width, label='LAET', edgecolor='black',zorder=3)
b3 = ax1.bar(indexes[2], ave_lat_faiss, width, label='Faiss', edgecolor='black',zorder=3)

ax1.set_ylim(bottom=0, top=300)

ax1.set_ylabel('Avg. Latency (ms)')
x_ticks = [0.2*i+0.12+width for i in range(num_datasets)]
y_ticks = [i for i in range(0, 350, 50)]
ax1.set_xticks(x_ticks)
ax1.set_yticks(y_ticks)
# ax1.get_yaxis().set_tick_params(pad=12)
ax1.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.05), prop={'size': font_size})

x_label = 'Dataset'
x_ticklabels = datasets.copy()
ax1.set_xticklabels(x_ticklabels)
# ax1.grid(axis='y', linestyle='--')

# Save the figure
file_path = '12-1.pdf'
plt.savefig(file_path, bbox_inches='tight', transparent=True, backend='pgf')