
import os
from string import printable
import sys
import time
import numpy as np
import re
import pickle
import argparse
import math
from multiprocessing.dummy import Pool as ThreadPool

import faiss
import util

#################################################################
# Bookkeeping
#################################################################
# Where the dataset base, query, learn files are stored.
DB_DIR = '/workspace/data/'
# Where the ground truth files are stored.
GT_DIR = 'ground_truth/'

LA_DIR = 'latency/'
# Where the *_trained.index files are stored.
TRAINED_IDX_DIR = 'trained_index/'
# Where the *_populated.index files and cluster indices are stored.
# NOTE that the *_populated.index files can be as large as tens of GBs.
POPULATED_IDX_DIR = 'populated_index/'
# Where the trained prediction model and training logs are stored.
MODEL_DIR = 'training_model/'
# Where the training and testing data files are stored.
TRAINING_DIR = 'training_data/'

if not os.path.isdir(POPULATED_IDX_DIR):
    print ("%s does not exist, creating it" % POPULATED_IDX_DIR)
    os.mkdir(POPULATED_IDX_DIR)

if not os.path.isdir(TRAINING_DIR):
    print ("%s does not exist, creating it" % TRAINING_DIR)
    os.mkdir(TRAINING_DIR)

parser = argparse.ArgumentParser(description='learned termination benchmark')
parser.add_argument('-mode', '--searchmode', help='search mode', required=True)
parser.add_argument('-batch', '--batchsize', help='batch size', default='1')
parser.add_argument('-train', '--trainsize', help='train size', default='0')
parser.add_argument('-cluster', '--numcluster',
    help='number of clusters', default='1')
parser.add_argument('-thread', '--numthread',
    help='number of threads', default='1')
parser.add_argument('-thresh', '--predthresh',
    help='prediction thresholds', default='1')
parser.add_argument('-bsearch', '--binarysearch',
    help='binary search parameters', default='0,0,0')
parser.add_argument('-db', '--dbname', help='database name', required=True)
parser.add_argument('-idx', '--indexkey', help='index key', required=True)
parser.add_argument('-param', '--parametersets',
    help='parameter sets', required=True, nargs='+')
parser.add_argument('-k', '--topk', help='the topk result', required=True)
parser.add_argument('-err', '--error', help='the requested error bound', required=True)
args = vars(parser.parse_args())

# -2 = generate training data.
# -1 = generate testing data.
# 0 = baseline (fixed nprobe or fixed efSearch).
# 1 = decision tree-based early termination.
search_mode = int(args['searchmode'])
batch_size = int(args['batchsize']) # batch size
train_size = int(args['trainsize']) # how many training vectors (in millions)
num_cluster = int(args['numcluster']) # number of cluster for IVF index
num_thread = int(args['numthread'])
input_k = int(args['topk'])
error_bound = int(args['error'])
# When to make prediction/generate training/testing data during search.
# This is related to the intermediate search result features.
pred_thresh = [int(x) for x in args['predthresh'].split(',')]
# Binary search to find minimum fixed configuration (for baseline) or minimum
# prediction multiplier (for early termination) to reach a certain accuracy
# target.
binary_search = int(args['binarysearch'].split(',')[0])
binary_range = [int(args['binarysearch'].split(',')[1]),
                int(args['binarysearch'].split(',')[2])]
dbname = args['dbname'] # e.g.: SIFT1M
index_key = args['indexkey'] # e.g.: IVF1000
parametersets = args['parametersets'] # e.g.: nprobe={1,2}

# Number of iterations over all queries (to get stable performance number).
num_iter = 1 
# When multi-threading is enabled, it indicates that latency measurement
# is not the purpose of the experiment. Thus we only run one iteration.
if num_thread > 1:
    num_iter = 1
print("Thread:", num_thread)
#################################################################
# Prepare dataset
#################################################################
print('Preparing dataset {}'.format(dbname))

if dbname.startswith('SIFT'):
    dbsize = int(dbname[4:-1])
    xb = util.mmap_fvecs('{}sift/sift10M/sift10M.fvecs'.format(DB_DIR))
    xq = util.mmap_fvecs('{}sift/sift10M/query.fvecs'.format(DB_DIR))
    xt = util.mmap_fvecs('{}sift/sift10M/sift10M.fvecs'.format(DB_DIR))
    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]
    print('DB size:', dbsize)
    gt = util.read_tsv('{}gtSIFT{}Mtest.tsv'.format(GT_DIR, dbsize))
    if search_mode == 0 and train_size > 0 and binary_search == 1:
        # Take a sample from the training vector to find the minimum fixed
        # termination condition to reach different accuracy targets. This is
        # needed to choose the intermediate search result features when
        # generating training data.
        xq = xt[:10000]
        gt = util.read_tsv('{}gtSIFT{}Mtrain{}M.tsv'.format(GT_DIR, dbsize,
            train_size))[:10000]
    if search_mode == -2:
        # xq = xt[:train_size * 1000 * 1000]
        # gt = util.read_tsv('{}gtSIFT{}Mtrain{}M.tsv'.format(GT_DIR, dbsize,
        #     train_size))
        # Change train data to query data by zzl
        xq = xq[:5000]
        gt = gt[:5000]
elif dbname.startswith('DEEP'):
    dbsize = int(dbname[4:-1])
    xb = util.mmap_fvecs('{}deep/deep10M.fvecs'.format(DB_DIR))
    xq = util.mmap_fvecs('{}deep/query.fvecs'.format(DB_DIR))
    xt = util.mmap_fvecs('{}deep/deep10M.fvecs'.format(DB_DIR))
    # trim xb to correct size
    xb = xb[:dbsize * 1000 * 1000]
    gt = util.read_tsv('{}gtDEEP{}Mtest.tsv'.format(GT_DIR, dbsize))
    if search_mode == 0 and train_size > 0 and binary_search == 1:
        # Take a sample from the training vector to find the minimum fixed
        # termination condition to reach different accuracy targets. This is
        # needed to choose the intermediate search result features when
        # generating training data.
        xq = xt[:10000]
        gt = util.read_tsv('{}gtDEEP{}Mtrain{}M.tsv'.format(GT_DIR, dbsize,
            train_size))[:10000]
    if search_mode == -2:
        # xq = xt[:train_size * 1000 * 1000]
        # gt = util.read_tsv('{}gtDEEP{}Mtrain{}M.tsv'.format(GT_DIR, dbsize,
        #     train_size))
        # Change train data to query data by zzl
        xq = xq[:5000]
        gt = gt[:5000]
elif dbname.startswith('GIST'):
    xb = util.mmap_fvecs('{}gist/gist1M.fvecs'.format(DB_DIR))
    xq = util.mmap_fvecs('{}gist/query.fvecs'.format(DB_DIR))
    xt = util.mmap_fvecs('{}gist/gist1M.fvecs'.format(DB_DIR))
    gt = util.read_tsv('{}gtGIST1Mtest.tsv'.format(GT_DIR))
    if search_mode == 0 and train_size > 0 and binary_search == 1:
        # Take a sample from the training vector to find the minimum fixed
        # termination condition to reach different accuracy targets. This is
        # needed to choose the intermediate search result features when
        # generating training data.
        xq = xt[:10000]
        gt = util.read_tsv('{}gtGIST1Mtrain500K.tsv'.format(GT_DIR))[:10000]
    if search_mode == -2:
        # xq = xt
        # gt = util.read_tsv('{}gtGIST1Mtrain500K.tsv'.format(GT_DIR))
         # Change train data to query data by zzl
        xq = xq[:500]
        gt = gt[:500]
elif dbname.startswith('SPACEV'):
    xb = util.mmap_fvecs('{}spacev/spacev10M.fvecs'.format(DB_DIR))
    xq = util.mmap_fvecs('{}spacev/query.fvecs'.format(DB_DIR))
    xt = util.mmap_fvecs('{}spacev/spacev10M.fvecs'.format(DB_DIR))
    gt = util.read_tsv('{}gtSPACEV10Mtest.tsv'.format(GT_DIR))
    if search_mode == -2:
        # xq = xt
        # gt = util.read_tsv('{}gtGIST1Mtrain500K.tsv'.format(GT_DIR))
         # Change train data to query data by zzl
        xq = xq[:5000]
        gt = gt[:5000]
elif dbname.startswith('GLOVE'):
    xb = util.mmap_fvecs('{}glove/glove.fvecs'.format(DB_DIR))
    xq = util.mmap_fvecs('{}glove/query.fvecs'.format(DB_DIR))
    xt = util.mmap_fvecs('{}glove/glove.fvecs'.format(DB_DIR))
    gt = util.read_tsv('{}gtGLOVE1Mtest.tsv'.format(GT_DIR))
    if search_mode == -2:
        # xq = xt
        # gt = util.read_tsv('{}gtGIST1Mtrain500K.tsv'.format(GT_DIR))
         # Change train data to query data by zzl
        xq = xq[:5000]
        gt = gt[:5000]
elif dbname.startswith('TEXT'):
    xb = util.mmap_fvecs('{}text/text10M.fvecs'.format(DB_DIR))
    xq = util.mmap_fvecs('{}text/query.fvecs'.format(DB_DIR))
    xt = util.mmap_fvecs('{}text/text10M.fvecs'.format(DB_DIR))
    gt = util.read_tsv('{}gtTEXT10Mtest.tsv'.format(GT_DIR))
    if search_mode == -2:
        # xq = xt
        # gt = util.read_tsv('{}gtGIST1Mtrain500K.tsv'.format(GT_DIR))
         # Change train data to query data by zzl
        xq = xq[:5000]
        gt = gt[:5000]
else:
    print(sys.stderr, 'unknown dataset', dbname)
    sys.exit(1)

print("sizes: B {} Q {} T {} gt {}".format(xb.shape, xq.shape, xt.shape,
    len(gt)),'*',len(gt[0]))
nq, d = xq.shape
nb, d = xb.shape

#################################################################
# Training
#################################################################
def choose_train_size(index_key):
    # some training vectors for PQ and the PCA
    n_train = 256 * 1000
    if "IVF" in index_key:
        matches = re.findall('IVF([0-9]+)', index_key)
        ncentroids = int(matches[0])
        n_train = max(n_train, 100 * ncentroids)
    elif "IMI" in index_key:
        matches = re.findall('IMI2x([0-9]+)', index_key)
        nbit = int(matches[0])
        n_train = max(n_train, 256 * (1 << nbit))
    return n_train

def get_trained_index():
    filename = "%s%s_%s_trained.index" % (
        TRAINED_IDX_DIR, dbname, index_key)
    if not os.path.exists(filename):
        index = faiss.index_factory(d, index_key)
        n_train = choose_train_size(index_key)
        # n_train = 256000
        print("n_train:", n_train)
        xtsub = xt[:n_train]
        print("Keeping {} train vectors".format(xtsub.shape[0]))
        # make sure the data is actually in RAM and in float
        xtsub = xtsub.astype('float32').copy()
        index.verbose = True

        t0 = time.time()
        index.train(xtsub)
        index.verbose = False
        print("train done in {} s".format(time.time() - t0))
        print("storing {}".format(filename))
        faiss.write_index(index, filename)
    else:
        print("loading {}".format(filename))
        index = faiss.read_index(filename)
    return index

#################################################################
# Adding vectors to dataset
#################################################################
def rate_limited_imap(f, l):
    'a thread pre-processes the next element'
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i[0], i[1], ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()

def matrix_slice_iterator(x, bs):
    " iterate over the lines of x in blocks of size bs"
    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]
    return rate_limited_imap(
        lambda i0, i1: x[i0:i1].astype('float32').copy(),
        block_ranges)

def get_populated_index():
    filename = "%s%s_%s_populated.index" % (
        POPULATED_IDX_DIR, dbname, index_key)
    filenameC = "%s%s_C%d" % (
        POPULATED_IDX_DIR, dbname, num_cluster)

    if not os.path.exists(filename):
        index = get_trained_index()
        index.verbose = True
        i0 = 0
        t0 = time.time()
        for xs in matrix_slice_iterator(xb, 100000):
            i1 = i0 + xs.shape[0]
            print('\radd {}:{}, {} s'.format(i0, i1, time.time() - t0)),
            sys.stdout.flush()
            index.add(xs)
            i0 = i1
        print
        print("Add done in {} s".format(time.time() - t0))
        print("storing {}".format(filename))
        index.verbose = False
        faiss.write_index(index, filename)
        faiss.write_cluster_id(index, filenameC)   
    else:
        print("loading {}".format(filename))
        index = faiss.read_index(filename)
    return index

#################################################################
# Perform searches
#################################################################
def compute_recall(result, ground, ncandidate):
    count = 0
    for i in range(len(result)):
        for j in range(ncandidate):
            if result[i][j] in ground[i]:
                count += 1
                break
    return count

# Binary search to find minimum fixed configuration (for baseline) or minimum
# prediction multiplier (for early termination) to reach a certain accuracy
# target.
def find_config(target, d, k):
    ret = float('inf')
    l, r = binary_range[0], binary_range[1]
    for key, value in d.items():
        if value < target:
            l = max(l, key+1)
        else:
            r = min(r, key)
    while l <= r:
        mid = l + int(math.floor((r-l)/2))
        # change the config to mid
        if index_key[:4] == 'HNSW':
            pa = 'efSearch={}'.format(mid)
        else:
            pa = 'nprobe={}'.format(mid)
        sys.stdout.flush()
        ps.set_index_parameters(index, pa)
        totalR = 0.0
        for i in range(0, nq, batch_size):
            query = xq[i:i+batch_size,:]
            D, I = index.search(query, k)
            totalR += compute_recall(I[:, :100], gt[i:i+batch_size], 100)
        totalR = totalR / float(nq)
        d[mid] = totalR
        print('{}, accuracy = {}'.format(pa, totalR))
        if totalR >= target:
            ret = min(ret, mid)
            r = mid-1
        else:
            l = mid+1
    if ret != float('inf'):
        return ret, target
    else:
        return find_config(max(d.values()), d, k)

# Read (and build if necessary) the search index.
index = get_populated_index()

# Load the prediction model for HNSW index.
if search_mode == 1 and index_key[:4] == 'HNSW':
    for t in pred_thresh:
        modelname = '{}{}_{}_model_thresh{}_Log_Full.txt'.format(MODEL_DIR, dbname,
            index_key, t)
        faiss.load_model(index, modelname)

# Load the prediction model for IVF index.
if search_mode == 1 and index_key[:4] != 'HNSW':
    for t in pred_thresh:
        modelname = '{}{}_{}_model_thresh{}_Full.txt'.format(MODEL_DIR, dbname,
            index_key, t)
        if index_key[:3] == 'OPQ' and int(dbname[4:-1]) == 1000:
            modelname = '{}{}_{}_model_thresh{}_Log_Full.txt'.format(MODEL_DIR, dbname,
                index_key, t)
        faiss.load_model(index, modelname)

# Load the pred_thresh into the search index.
if search_mode != 0:
    faiss.load_thresh(index, -1)
    for t in pred_thresh:
        faiss.load_thresh(index, t)

ps = faiss.ParameterSpace()
ps.initialize(index)
print("ParameterSpace initialize done")

# make sure queries are in RAM
xq = xq.astype('float32').copy()

# Where the training/testing data will be stored before written to files.
if search_mode < 0:
    if index_key[:4] == 'HNSW':
        data = []
    else:
        data = []

param_list = []
for param in parametersets:
    param_list.append(param)
recall_list = [0.0]*len(parametersets)
latency_list = [0.0]*len(parametersets)

faiss.omp_set_num_threads(num_thread)
k = input_k

# To get the cluster indices where the ground truth nearest neighbor(s) reside
# for the IVF case. We need this to determine the minimum termination
# condition. This is achieved by combining two things: 1) Using the
# write_cluster_id() we wrote the cluster index of all database vectors into
# files. 2) In a computeGT.py we performed exhaustive search to find which
# database vectors are ground truth nearest neighbor(s).
if search_mode < 0 and index_key[:4] != 'HNSW':
    if index_key[:3] == 'OPQ':
        pkl_filename = '{}{}_C{}_gtcluster{}_opq.pkl'.format(GT_DIR, dbname,
            num_cluster, -1*search_mode)
    else:
        pkl_filename = '{}{}_C{}_gtcluster{}.pkl'.format(GT_DIR, dbname,
            num_cluster, -1*search_mode)
    if not os.path.exists(pkl_filename):
        if index_key[:3] == 'OPQ':
            clusterid = (np.fromfile('{}{}_C{}_clusterid_quantized.tsv'.format(
                POPULATED_IDX_DIR, dbname, num_cluster), dtype='uint64',
                sep='\t')).reshape(-1, 2)
        else:
            clusterid = (np.fromfile('{}/{}_C{}_clusterid.tsv'.format(
                POPULATED_IDX_DIR, dbname, num_cluster), dtype='uint64',
                sep='\t')).reshape(-1, 2)
        gt_clusters = {}
        # If a database vector is a ground truth nearest neighbor to a query,
        # insert its index as a key into gt_clusters dict.
        for i in range(nq):
            for j in range(len(gt[i])):
                gt_clusters[gt[i][j]] = 0
        # Then for each key, insert its cluster index as the value into
        # gt_clusters dict.
        for i in range(len(clusterid)):
            if clusterid[i][0] in gt_clusters:
                gt_clusters[clusterid[i][0]] = clusterid[i][1]
        output = open(pkl_filename, 'wb')
        pickle.dump(gt_clusters, output)
        output.close()
    else:
        pkl_file = open(pkl_filename, 'rb')
        gt_clusters = pickle.load(pkl_file)
        pkl_file.close()

if binary_search == 0:
    for it in range(num_iter):
        print('iteration {}'.format(it))
        print("paramtersets:", parametersets);
        if k < 10:
            print(' '*(len(parametersets[-1])+1)+'R@1    R@10   R@100  time(ms)')
        elif k < 100:
            print(' '*(len(parametersets[-1])+1)+'R@1    R@10   R@100  time(ms)')
        else:
            print(' '*(len(parametersets[-1])+1)+'R@1    R@10   R@100  time(ms)')
        fi = open('./result/recall.log','w')
        for param in range(len(parametersets)):
            sys.stdout.flush()
            ps.set_index_parameters(index, parametersets[param])
            total_recall_at1 = 0.0
            total_recall_at10 = 0.0
            total_recall_at100 = 0.0
            total_latency = 0.0
            fnn = dbname[:4]
            fnn = fnn.lower()
            if fnn == 'sift' or fnn == 'deep':
                fnn += '10M'
            latency_file = 'LAET_Latency_'+fnn+'_'+str(input_k)+'_'+str(error_bound)+'.log'
            la_f = open(latency_file, 'w')
            for i in range(0, nq, batch_size):
                # When generating training/testing data for the IVF case,
                # load the cluster indices where the ground truth nearest
                # neighbor(s) reside.
                if search_mode < 0 and index_key[:4] != 'HNSW':
                    faiss.load_gt(index, -1)
                    for j in range(batch_size):
                        faiss.load_gt(index, -2)
                        for l in range(len(gt[i+j])):
                            faiss.load_gt(index, int(gt_clusters[gt[i+j][l]]))
                # When generating training/testing data for the HNSW case,
                # load the database vector indices of the ground truth
                # nearest neighbor(s).
                if search_mode < 0 and index_key[:4] == 'HNSW':
                    faiss.load_gt(index, -1)
                    for j in range(batch_size):
                        faiss.load_gt(index, -2)
                        for l in range(len(gt[i+j])):
                            faiss.load_gt(index, int(gt[i+j][l]))
                query = xq[i:i+batch_size,:]
                t0 = time.time()
                D, I = index.search(query, k)
                t1 = time.time()
                cur_latency = t1 - t0
                if i >= nq // 2:
                    print(cur_latency, file=la_f)
                total_latency += t1-t0
                # total_recall_at1 += compute_recall(I[:, :1],
                #     gt[i:i+batch_size], 1)
                # total_recall_at10 += compute_recall(I[:, :10],
                #     gt[i:i+batch_size], 10)
                # total_recall_at100 += compute_recall(I[:, :100],
                #     gt[i:i+batch_size], 100)
                # print(compute_recall(I[:, :100],
                #     gt[i:i+batch_size], 100), file=fi)
                if search_mode < 0:
                    # When generating training/testing data, read the returned
                    # search results (since this is where we stored the
                    # features and targe values).
                    if index_key[:4] == 'HNSW':
                        for j in range(len(I)):
                            line = []
                            line.append(int(D[j][0]))
                            line.append(i+j)
                            for l in range(1+4*len(pred_thresh)):
                                line.append(D[j][l+1])
                            data.append(line)
                    else:
                        for j in range(len(I)):
                            line = []
                            line.append(int(D[j][0]))
                            line.append(i+j)
                            for l in range(10+4*len(pred_thresh)):
                                line.append(D[j][l+1])
                            data.append(line)
            la_f.close()
            tr1 = total_recall_at1 / float(nq)
            tr10 = total_recall_at10 / float(nq)
            tr100 = total_recall_at100 / float(nq)
            tt = total_latency * 1000.0 / nq
            print(parametersets[param]+
                ' '*(len(parametersets[-1])+1-len(parametersets[param]))+
                '{:.4f} {:.4f} {:.4f} {:.4f}'.format(
                round(tr1,4), round(tr10,4), round(tr100,4), round(tt,4)))
            if it > 0 or num_iter == 1:
                recall_list[param] += total_recall_at100
                latency_list[param] += total_latency
    # Write the training/testing data files.
    if search_mode == -1:
        util.write_tsv(data, '{}{}_{}_test.tsv'.format(TRAINING_DIR, dbname,
            index_key))
    if search_mode == -2:
        util.write_tsv(data, '{}{}_{}_train.tsv'.format(TRAINING_DIR, dbname,
            index_key))

    denom = float(nq*max(num_iter-1, 1))
    recall_list = [x/denom for x in recall_list]
    latency_list = [round(x*1000.0/denom, 4) for x in latency_list]

    print('param_list = {}'.format(param_list))
    print('recall target = {}'.format(recall_list))
    print('average latency(ms) = {}'.format(latency_list))
    print('result_{}_{} = {}'.format(dbname, index_key, [latency_list, recall_list]))
else:
    # Binary search to find minimum fixed configuration (for baseline) or minimum
    # prediction multiplier (for early termination) to reach a certain accuracy
    # target.
    target = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
    if dbname.startswith('DEEP10M') and index_key[:4] == 'HNSW':
        target = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.9955]
    if dbname.startswith('GIST1M') and index_key[:4] == 'HNSW':
        target = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]
    # For billion-scale, stop at 0.995 because it takes too long to reach 1.0.
    if int(dbname[4:-1]) == 1000:
        target = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
    res = []
    d = {}
    sys.stdout.flush()
    ps.set_index_parameters(index, parametersets[0])
    for t in target:
        ret, act_t = find_config(t, d, k)
        print('To reach recall target {} the min. config/multiplier is {}.'.format(act_t, ret))
        res.append(ret)
        if act_t != t:
            break
    print('List of min. config/multiplier = {}'.format(res))

