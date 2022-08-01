
import time
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

import faiss
import util

def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(1)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i[0], i[1], ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()
    pool.close()
    pool.join()

class IdentPreproc:
    """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x

def sanitize(x):
    """ convert array to a c-contiguous float array """
    # return np.ascontiguousarray(x.astype('float32'))
    return np.ascontiguousarray(x, dtype='float32')

def dataset_iterator(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    # def prepare_block((i0, i1)):
    def prepare_block(i0, i1):
        xb = sanitize(x[i0:i1])
        return i0, preproc.apply_py(xb)

    return rate_limited_imap(prepare_block, block_ranges)

# This is the modified CPU version of compute_GT from Faiss.
# Performs exhaustive search to find ground truth nearest neighbors.
def compute_GT_CPU(xb, xq, gt_sl):
    nq_gt, _ = xq.shape
    print("compute GT CPU")
    t0 = time.time()

    gt_I = np.zeros((nq_gt, gt_sl), dtype='int64')
    gt_D = np.zeros((nq_gt, gt_sl), dtype='float32')
    heaps = faiss.float_maxheap_array_t()
    heaps.k = gt_sl
    heaps.nh = nq_gt
    heaps.val = faiss.swig_ptr(gt_D)
    heaps.ids = faiss.swig_ptr(gt_I)
    heaps.heapify()
    bs = 10 ** 5

    n, d = xb.shape
    xqs = sanitize(xq[:nq_gt])

    db_gt = faiss.IndexFlatL2(d)

    # compute ground-truth by blocks of bs, and add to heaps
    for i0, xsl in dataset_iterator(xb, IdentPreproc(d), bs):
        db_gt.add(xsl)
        D, I = db_gt.search(xqs, gt_sl)
        I += i0
        heaps.addn_with_ids(
            gt_sl, faiss.swig_ptr(D), faiss.swig_ptr(I), gt_sl)
        db_gt.reset()
    heaps.reorder()

    print("GT CPU time: {} s".format(time.time() - t0))
    return gt_I, gt_D

# This is the modified GPU version of compute_GT from Faiss.
# Performs exhaustive search to find ground truth nearest neighbors.
def compute_GT_GPU(xb, xq, gt_sl):
    nq_gt, _ = xq.shape
    print("compute GT GPU")
    t0 = time.time()

    gt_I = np.zeros((nq_gt, gt_sl), dtype='int64')
    gt_D = np.zeros((nq_gt, gt_sl), dtype='float32')
    heaps = faiss.float_maxheap_array_t()
    heaps.k = gt_sl
    heaps.nh = nq_gt
    heaps.val = faiss.swig_ptr(gt_D)
    heaps.ids = faiss.swig_ptr(gt_I)
    heaps.heapify()
    bs = 10 ** 5
    # Please change this based on your GPU memory size.
    tempmem = 3500*1024*1024

    n, d = xb.shape
    xqs = sanitize(xq[:nq_gt])
 
    ngpu = faiss.get_num_gpus()
    gpu_resources = []

    for i in range(ngpu):
        res = faiss.StandardGpuResources()
        res.setTempMemory(tempmem)
        gpu_resources.append(res)

    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    for i in range(0, ngpu):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])

    db_gt = faiss.IndexFlatL2(d)
    db_gt_gpu = faiss.index_cpu_to_gpu_multiple(
        vres, vdev, db_gt)

    # compute ground-truth by blocks of bs, and add to heaps
    for i0, xsl in dataset_iterator(xb, IdentPreproc(d), bs):
        db_gt_gpu.add(xsl)
        D, I = db_gt_gpu.search(xqs, gt_sl)
        I += i0
        heaps.addn_with_ids(
            gt_sl, faiss.swig_ptr(D), faiss.swig_ptr(I), gt_sl)
        db_gt_gpu.reset()
    heaps.reorder()

    print("GT GPU time: {} s".format(time.time() - t0))
    return gt_I, gt_D

# Compute the ground truth nearest neighbor(s) database indexes for both
# training and query vectors. Note that it requires the GPU version of Faiss
# to increase the speed: essentially we are doing exhaustive search, so CPU
# version is too slow when the database/number of training vectors is large.
# 
# For the GPU version of Faiss, we recommend NOT to build from our github
# implementation, but install via Conda on a different machine. This is because
# building the GPU version from source is much more difficult than CPU version.
# Or you can just simply use the ground truth we provide in the github repo.
#
# One difference between our function and the public ground truth dataset is
# that we include allow tied ground truth nearest neighors (i.e. for a single
# query there might be multiple ground truth database vectors). We found ties
# happen when e.g. there are duplicated database vectors (which is true for all
# 3 datasets we used).
#
# Inputs:
# dataset: name of the dataset.
# nb: the database size.
# nt: the number of training vectors.
# k: how many top nearest neighbors to evaluate when finding the ground truth
# nearest neighbors. We need a high enough k to cover any possible tied ground
# truth neighbors.
# input_dir: where the base, query, learn files are stored.
# output_dir: where the ground truth output files will be written.
# train: whether we are generating ground truth for training or query vectors.
# cpu: whether or not use CPU to perform the exhaustive search.
def compute_GT(dataset, nb, nt, k, input_dir, output_dir, train=False,
    cpu=False):
    if dataset == 'DEEP':
        xb = util.mmap_fvecs('{}deep1B_base.fvecs'.format(
            input_dir))[:nb*1000000]
        xq = util.mmap_fvecs('{}deep1B_query.fvecs'.format(input_dir))
        xt = util.mmap_fvecs('{}deep1B_learn.fvecs'.format(
            input_dir))[:nt*1000000]
    elif dataset == 'SIFT':
        xb = util.mmap_bvecs('{}bigann_base.bvecs'.format(
            input_dir))[:nb*1000000]
        xq = util.mmap_bvecs('{}bigann_query.bvecs'.format(input_dir))
        xt = util.mmap_bvecs('{}bigann_learn.bvecs'.format(
            input_dir))[:nt*1000000]
    elif dataset == 'GIST':
        # For GIST we don't use the nb and nt, since we always use all the
        # database and training vectors.
        xb = util.mmap_fvecs('{}gist_base.fvecs'.format(input_dir))
        xq = util.mmap_fvecs('{}gist_query.fvecs'.format(input_dir))
        xt = util.mmap_fvecs('{}gist_learn.fvecs'.format(input_dir))
    if train:
        data = []
        # Split the training vectors into 1 million chunks to reduce memory
        # footprint.
        for i_t in range(nt):
            if cpu:
                gt_I, gt_D = compute_GT_CPU(xb,
                    xt[i_t*1000000:(i_t+1)*1000000], k)
            else:
                gt_I, gt_D = compute_GT_GPU(xb,
                    xt[i_t*1000000:(i_t+1)*1000000], k)
            for i in range(len(gt_I)):
                candidate = []
                for j in range(k):
                    if gt_D[i][j] ==  gt_D[i][0]:
                        candidate.append(gt_I[i][j])
                data.append(candidate)
        if dataset == 'GIST':
            util.write_tsv(data, '{}gtGIST1Mtrain500K.tsv'.format(output_dir))
        else:
            util.write_tsv(data, '{}gt{}{}Mtrain{}M.tsv'.format(output_dir,
                dataset, nb, nt))
    else:
        if cpu:
            gt_I, gt_D = compute_GT_CPU(xb, xq, k)
        else:
            gt_I, gt_D = compute_GT_GPU(xb, xq, k)
        data = []
        for i in range(len(gt_I)):
            candidate = []
            for j in range(k):
                if gt_D[i][j] ==  gt_D[i][0]:
                    candidate.append(gt_I[i][j])
            data.append(candidate)
        util.write_tsv(data, '{}gt{}{}Mtest.tsv'.format(output_dir, dataset,
            nb))

if __name__ == "__main__":
    # Where the dataset base, query, learn files are stored.
    INPUT_DIR = '/mnt/hdd/conglonl/'
    # Where the ground truth output files will be written.
    OUTPUT_DIR = '/mnt/hdd/conglonl/'

    # We used CPU version for this single case because we found that for one of
    # the query (6680th), the ground truth computed by the CPU version (16517
    # and 8271565) and the GPU version (16517) are different. Since in
    # performance evaluation we aim to achieve 100% accuracy in CPU, we have to
    # compute the ground truth in CPU for this case. We didn't have time to
    # find the root cause of this problem, but it is probably related to the
    # precision of the distance value.
    compute_GT('DEEP', 10, 1, 1000, INPUT_DIR, OUTPUT_DIR, train=False, cpu=True)

    compute_GT('DEEP', 10, 1, 100, INPUT_DIR, OUTPUT_DIR, train=True)
    compute_GT('SIFT', 10, 1, 1000, INPUT_DIR, OUTPUT_DIR, train=False)
    compute_GT('SIFT', 10, 1, 100, INPUT_DIR, OUTPUT_DIR, train=True)
    compute_GT('GIST', 1, 1, 1000, INPUT_DIR, OUTPUT_DIR, train=False)
    compute_GT('GIST', 1, 1, 100, INPUT_DIR, OUTPUT_DIR, train=True)

    compute_GT('DEEP', 1000, 1, 1000, INPUT_DIR, OUTPUT_DIR, train=False)
    compute_GT('DEEP', 1000, 1, 100, INPUT_DIR, OUTPUT_DIR, train=True)
    compute_GT('SIFT', 1000, 1, 1000, INPUT_DIR, OUTPUT_DIR, train=False)
    compute_GT('SIFT', 1000, 1, 100, INPUT_DIR, OUTPUT_DIR, train=True)

