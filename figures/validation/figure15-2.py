import os
import sys
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import palettable
import scipy as sp
from scipy.optimize import leastsq

stop_list = ['', '\n']

def slide(a):
    slide_window = 7
    width = slide_window // 2
    b = a.copy()
    for i in range(width, len(a)-width):
        b[i] = sum(a[i-width: i+width+1]) / (2*width+1)
    return b

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

# Set font and figure size
font_size = 30
plt.rc('font',**{'size': font_size, 'family': 'Arial' })
plt.rc('pdf',fonttype = 42)
fig_size = (9, 6)
fig, ax2 = plt.subplots(figsize=fig_size)
# matplotlib.rcParams['xtick.major.size'] = 8.
# matplotlib.rcParams['xtick.minor.size'] = 4.
# matplotlib.rcParams['ytick.major.size'] = 8.
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)

n_probes = [1]
k = [[] for i in range(len(n_probes))]
v = [[] for i in range(len(n_probes))]
for j in range(len(n_probes)):
    n = n_probes[j]
    file_name = '../../Auncel/eval/Validation_96_'+str(n)+'.log'
    res = read_float(file_name)
    for i in res:
        k[j].append(i[0])
        v[j].append(i[1])

for i in range(len(n_probes)):
#     v[i] = slide(v[i])
    for j in range(len(v[i])):
        v[i][j] = 1.0 * v[i][j]

k_sample = []
v_sample = []
d = len(k[0]) // 150
i = 0
while i*d < len(k[0]):
    if i*(d+1) < len(k[0]):
        tmp_k = k[0][i*d: (i+1)*d]
        tmp_v = v[0][i*d: (i+1)*d]
    else:
        tmp_k = k[0][i*d: len(k[0])]
        tmp_v = v[0][i*d: len(k[0])]
    i = i + 1
    max_j = 0
    for j in range(1, len(tmp_v)):
        if tmp_v[max_j] < tmp_v[j]:
            max_j = j
    k_sample.append(tmp_k[max_j])
    v_sample.append(tmp_v[max_j])
      

def func(p,x):
    a, b=p
    return 1.0 / (a*x+b)

def error(p,x,y):
    return func(p,x)-y

p0=[-0.02, 0.5]

# Xi = np.array(k[0])
# Yi = np.array(v[0])
Xi = np.array(k_sample)
Yi = np.array(v_sample)
Para=leastsq(error,p0,args=(Xi,Yi))

a, b=Para[0]
# print("a=",a, "b=", b)
# print("cost："+str(Para[1]))

x=np.linspace(0,22,80)
y=1.0 / (a*x+b)
ax2.plot(x,y,color="red",label=r"$\varphi$" + " Upper \nBound Model", linewidth=3, zorder=3)
ax2.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.06), prop={'size': font_size})

#####################################################################
# ax2
#####################################################################

# Scatter-plot of CPU-time vs Persistent data exchanged
def ScatterPlot(x, y):
    x_bins = np.linspace(0, 25, 80)
    y_bins = np.linspace(0, 15, 80)
    
    Z, xedges, yedges = np.histogram2d(x, y, [x_bins,y_bins])
#     Z, xedges, yedges = np.histogram2d(x, y)
    p = ax2.pcolormesh(xedges, yedges, Z.T, norm=colors.LogNorm(vmin=1, vmax=Z.max()), cmap='Greens')
#     p = ax2.pcolormesh(xedges, yedges, Z.T, cmap='Greens')
    fig.colorbar(p)

#     ax2.set_yscale('log')
#     ax2.set_xscale('log')

# ax2.set_ylim((bottom=0, top=20)
ScatterPlot(k[0], v[0])
ax2.set_xlabel('Upper Bound Function ('+r'$U$'+')')
ax2.set_ylabel('Scaling Factor ('+r'$\varphi$'+')')

# Save the figure
file_path = '15-2.pdf'
plt.savefig(file_path, bbox_inches='tight', transparent=True)