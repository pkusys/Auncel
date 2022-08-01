#!/usr/bin/env python2

import csv
import numpy as np
import os

#################################################################
# I/O functions
#################################################################

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def mmap_vecs(fname, dimension, dtype):
    x = np.memmap(fname, dtype=dtype, mode='r')
    return x.reshape(-1, dimension)

def mmap_flat(fname, dtype):
    x = np.memmap(fname, dtype=dtype, mode='r')
    return x.ravel()

def get_stuff(filename, delimiter='\t'):
    with open(filename, 'rU') as tsvfile:
        datareader = csv.reader(tsvfile, delimiter=delimiter, dialect=csv.excel_tab)
        for row in datareader:
            yield row

def read_tsv(filename, delimiter='\t', doInt=False, multi=1, rnd=-1):
    ret = []
    for row in get_stuff(filename, delimiter):
        l = [float(x) for x in row]
        if multi != 1:
            l = [multi*x for x in l]
        if rnd != -1:
            l = [round(x, rnd) for x in l]
        if doInt:
            l = [int(x) for x in l]
        ret.append(l)
    return ret

def write_tsv(data, filename, delimiter='\t', append=False, doInt=False, multi=1, rnd=-1):
    if not append:
        try:
            os.remove(filename)
        except OSError:
            pass

    d = data
    if doInt or multi != 1 or rnd != -1:
        for i in range(len(d)):
            if multi != 1:
                d[i] = [multi*x for x in d[i]]
            if rnd != -1:
                d[i] = [round(x, rnd) for x in d[i]]
            if doInt:
                d[i] = [int(x) for x in d[i]]

    with open(filename, 'a') as tsvfile:
        writer = csv.writer(tsvfile, delimiter=delimiter)
        writer.writerows(d)

