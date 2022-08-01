/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexIVF.h"


#include <omp.h>

#include <cstdio>
#include <memory>
#include <fstream>

#include "utils.h"
#include "hamming.h"

#include "FaissAssert.h"
#include "IndexFlat.h"
#include "AuxIndexStructures.h"

namespace faiss {

using ScopedIds = InvertedLists::ScopedIds;
using ScopedCodes = InvertedLists::ScopedCodes;

/*****************************************
 * Level1Quantizer implementation
 ******************************************/


Level1Quantizer::Level1Quantizer (Index * quantizer, size_t nlist):
    quantizer (quantizer),
    nlist (nlist),
    quantizer_trains_alone (0),
    own_fields (false),
    clustering_index (nullptr)
{
    // here we set a low # iterations because this is typically used
    // for large clusterings (nb this is not used for the MultiIndex,
    // for which quantizer_trains_alone = true)
    cp.niter = 10;
}

Level1Quantizer::Level1Quantizer ():
    quantizer (nullptr),
    nlist (0),
    quantizer_trains_alone (0), own_fields (false),
    clustering_index (nullptr)
{}

Level1Quantizer::~Level1Quantizer ()
{
    if (own_fields) delete quantizer;
}

void Level1Quantizer::train_q1 (size_t n, const float *x, bool verbose, MetricType metric_type)
{
    size_t d = quantizer->d;
    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (verbose)
            printf ("IVF quantizer does not need training.\n");
    } else if (quantizer_trains_alone == 1) {
        if (verbose)
            printf ("IVF quantizer trains alone...\n");
        quantizer->train (n, x);
        quantizer->verbose = verbose;
        FAISS_THROW_IF_NOT_MSG (quantizer->ntotal == nlist,
                          "nlist not consistent with quantizer size");
    } else if (quantizer_trains_alone == 0) {
        if (verbose)
            printf ("Training level-1 quantizer on %ld vectors in %ldD\n",
                    n, d);

        Clustering clus (d, nlist, cp);
        quantizer->reset();
        if (clustering_index) {
            clus.train (n, x, *clustering_index);
            quantizer->add (nlist, clus.centroids.data());
        } else {
            clus.train (n, x, *quantizer);
        }
        quantizer->is_trained = true;
    } else if (quantizer_trains_alone == 2) {
        if (verbose)
            printf (
                "Training L2 quantizer on %ld vectors in %ldD%s\n",
                n, d,
                clustering_index ? "(user provided index)" : "");
        FAISS_THROW_IF_NOT (metric_type == METRIC_L2);
        Clustering clus (d, nlist, cp);
        if (!clustering_index) {
            IndexFlatL2 assigner (d);
            clus.train(n, x, assigner);
        } else {
            clus.train(n, x, *clustering_index);
        }
        if (verbose)
            printf ("Adding centroids to quantizer\n");
        quantizer->add (nlist, clus.centroids.data());
    }
}



/*****************************************
 * IndexIVF implementation
 ******************************************/


IndexIVF::IndexIVF (Index * quantizer, size_t d,
                    size_t nlist, size_t code_size,
                    MetricType metric):
    Index (d, metric),
    Level1Quantizer (quantizer, nlist),
    invlists (new ArrayInvertedLists (nlist, code_size)),
    own_invlists (true),
    code_size (code_size),
    nprobe (1),
    max_codes (0),
    parallel_mode (0),
    maintain_direct_map (false)
{
    FAISS_THROW_IF_NOT (d == quantizer->d);
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    // Spherical by default if the metric is inner_product
    if (metric_type == METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }

}

IndexIVF::IndexIVF ():
    invlists (nullptr), own_invlists (false),
    code_size (0),
    nprobe (1), max_codes (0), parallel_mode (0),
    maintain_direct_map (false)
{}

void IndexIVF::add (idx_t n, const float * x)
{
    add_with_ids (n, x, nullptr);
}


void IndexIVF::add_with_ids (idx_t n, const float * x, const long *xids)
{
    // do some blocking to avoid excessive allocs
    idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min (n, i0 + bs);
            if (verbose) {
                printf("   IndexIVF::add_with_ids %ld:%ld\n", i0, i1);
            }
            add_with_ids (i1 - i0, x + i0 * d,
                          xids ? xids + i0 : nullptr);
        }
        return;
    }

    FAISS_THROW_IF_NOT (is_trained);
    std::unique_ptr<idx_t []> idx(new idx_t[n]);
    quantizer->assign (n, x, idx.get());
    size_t nadd = 0, nminus1 = 0;

    for (size_t i = 0; i < n; i++) {
        if (idx[i] < 0) nminus1++;
    }

    std::unique_ptr<uint8_t []> flat_codes(new uint8_t [n * code_size]);
    encode_vectors (n, x, idx.get(), flat_codes.get());

#pragma omp parallel reduction(+: nadd)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            long list_no = idx [i];
            if (list_no >= 0 && list_no % nt == rank) {
                long id = xids ? xids[i] : ntotal + i;
                invlists->add_entry (list_no, id,
                                     flat_codes.get() + i * code_size);
                nadd++;
            }
        }
    }

    if (verbose) {
        printf("    added %ld / %ld vectors (%ld -1s)\n", nadd, n, nminus1);
    }

    ntotal += n;
}


void IndexIVF::make_direct_map (bool new_maintain_direct_map)
{
    // nothing to do
    if (new_maintain_direct_map == maintain_direct_map)
        return;

    if (new_maintain_direct_map) {
        direct_map.resize (ntotal, -1);
        for (size_t key = 0; key < nlist; key++) {
            size_t list_size = invlists->list_size (key);
            ScopedIds idlist (invlists, key);

            for (long ofs = 0; ofs < list_size; ofs++) {
                FAISS_THROW_IF_NOT_MSG (
                       0 <= idlist [ofs] && idlist[ofs] < ntotal,
                       "direct map supported only for seuquential ids");
                direct_map [idlist [ofs]] = key << 32 | ofs;
            }
        }
    } else {
        direct_map.clear ();
    }
    maintain_direct_map = new_maintain_direct_map;
}


void IndexIVF::search (idx_t n, const float *x, idx_t k,
                         float *distances, idx_t *labels) const
{
    if (search_mode == 0) { // original fixed configuration baseline
        long * idx = new long [n * nprobe];
        ScopeDeleter<long> del (idx);
        float * coarse_dis = new float [n * nprobe];
        ScopeDeleter<float> del2 (coarse_dis);

        double t0 = getmillisecs();
        quantizer->search (n, x, nprobe, coarse_dis, idx);
        indexIVF_stats.quantization_time += getmillisecs() - t0;

        t0 = getmillisecs();
        invlists->prefetch_lists (idx, n * nprobe);

        search_preassigned (n, x, k, idx, coarse_dis,
                            distances, labels, false);
        indexIVF_stats.search_time += getmillisecs() - t0;
    } else {
        long num_candidate_cluster;
        if (search_mode == 1 || search_mode == 2) {
            // Need at least 100 because it is required by the query-centroid
            // distance ratio features. If you have less than 100 total
            // clusters, you need to redefine the features and change
            // corresponding code.
            num_candidate_cluster = pred_max;
            num_candidate_cluster = std::max((long)100, num_candidate_cluster);
        } else if (search_mode == 3) {
            // For the simple heuristic-based approach for comparison we
            // assume that queries only need to search at most top 20% nearest
            // clusters since all test queries meet this assumption.
            num_candidate_cluster = invlists->nlist/5;
        } else {
            FAISS_THROW_MSG ("unsupported search_mode");
        }
        long * idx = new long [n * num_candidate_cluster];
        ScopeDeleter<long> del (idx);
        float * coarse_dis = new float [n * num_candidate_cluster];
        ScopeDeleter<float> del2 (coarse_dis);

        double t0 = getmillisecs();
        quantizer->search (n, x, num_candidate_cluster, coarse_dis, idx);
        indexIVF_stats.quantization_time += getmillisecs() - t0;

        t0 = getmillisecs();
        invlists->prefetch_lists (idx, n * num_candidate_cluster);

        search_preassigned_custom (n, x, k, idx, coarse_dis, distances,
                                   labels, false, num_candidate_cluster);
        indexIVF_stats.search_time += getmillisecs() - t0;
    }
}



void IndexIVF::search_preassigned (idx_t n, const float *x, idx_t k,
                                   const idx_t *keys,
                                   const float *coarse_dis ,
                                   float *distances, idx_t *labels,
                                   bool store_pairs,
                                   const IVFSearchParameters *params) const
{
    long nprobe = params ? params->nprobe : this->nprobe;
    long max_codes = params ? params->max_codes : this->max_codes;

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    idx_t check_period = InterruptCallback::get_period_hint
        (nprobe * ntotal * d / nlist);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel reduction(+: nlistv, ndis, nheap)
        {
            InvertedListScanner *scanner = get_InvertedListScanner(store_pairs);
            ScopeDeleter1<InvertedListScanner> del(scanner);

            /*****************************************************
             * Depending on parallel_mode, there are two possible ways
             * to organize the search. Here we define local functions
             * that are in common between the two
             ******************************************************/

            // intialize + reorder a result heap

            auto init_result = [&](float *simi, idx_t *idxi) {
                if (metric_type == METRIC_INNER_PRODUCT) {
                    heap_heapify<HeapForIP> (k, simi, idxi);
                } else {
                    heap_heapify<HeapForL2> (k, simi, idxi);
                }
            };

            auto reorder_result = [&] (float *simi, idx_t *idxi) {
                if (metric_type == METRIC_INNER_PRODUCT) {
                    heap_reorder<HeapForIP> (k, simi, idxi);
                } else {
                    heap_reorder<HeapForL2> (k, simi, idxi);
                }
            };

            // single list scan using the current scanner (with query
            // set porperly) and storing results in simi and idxi
            auto scan_one_list = [&] (idx_t key, float coarse_dis_i,
                                      float *simi, idx_t *idxi) {

                if (key < 0) {
                    // not enough centroids for multiprobe
                    return (size_t)0;
                }
                FAISS_THROW_IF_NOT_FMT (key < (idx_t) nlist,
                        "Invalid key=%ld nlist=%ld\n",
                        key, nlist);

                size_t list_size = invlists->list_size(key);

                // don't waste time on empty lists
                if (list_size == 0) {
                    return (size_t)0;
                }

                scanner->set_list (key, coarse_dis_i);

                nlistv++;

                InvertedLists::ScopedCodes scodes (invlists, key);

                std::unique_ptr<InvertedLists::ScopedIds> sids;
                const Index::idx_t * ids = nullptr;

                if (!store_pairs)  {
                    sids.reset (new InvertedLists::ScopedIds (invlists, key));
                    ids = sids->get();
                }

                nheap += scanner->scan_codes (list_size, scodes.get(),
                                              ids, simi, idxi, k);

                return list_size;
            };

            /****************************************************
             * Actual loops, depending on parallel_mode
             ****************************************************/

            if (parallel_mode == 0) {

#pragma omp for
                for (size_t i = i0; i < i1; i++) {
                    // loop over queries
                    scanner->set_query (x + i * d);
                    float * simi = distances + i * k;
                    idx_t * idxi = labels + i * k;

                    init_result (simi, idxi);

                    long nscan = 0;

                    // loop over probes
                    for (size_t ik = 0; ik < nprobe; ik++) {

                        nscan += scan_one_list (
                             keys [i * nprobe + ik],
                             coarse_dis[i * nprobe + ik],
                             simi, idxi
                        );

                        if (max_codes && nscan >= max_codes) {
                            break;
                        }
                    }

                    ndis += nscan;
                    reorder_result (simi, idxi);
                } // parallel for
            } else if (parallel_mode == 1) {
                std::vector <idx_t> local_idx (k);
                std::vector <float> local_dis (k);

                for (size_t i = i0; i < i1; i++) {
                    scanner->set_query (x + i * d);
                    init_result (local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                    for (size_t ik = 0; ik < nprobe; ik++) {
                        ndis += scan_one_list (
                            keys [i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            local_dis.data(), local_idx.data()
                        );

                        // can't do the test on max_codes
                    }
                    // merge thread-local results

                    float * simi = distances + i * k;
                    idx_t * idxi = labels + i * k;
#pragma omp single
                    init_result (simi, idxi);

#pragma omp barrier
#pragma omp critical
                    {
                        if (metric_type == METRIC_INNER_PRODUCT) {
                            heap_addn<HeapForIP>
                                (k, simi, idxi,
                                 local_dis.data(), local_idx.data(), k);
                        } else {
                            heap_addn<HeapForL2>
                                (k, simi, idxi,
                                 local_dis.data(), local_idx.data(), k);
                        }
                    }
#pragma omp barrier
#pragma omp single
                    reorder_result (simi, idxi);
                }
            } else {
                FAISS_THROW_FMT ("parallel_mode %d not supported\n",
                                 parallel_mode);
            }
        } // loop over blocks
        InterruptCallback::check ();
    } // loop over blocks

    indexIVF_stats.nq += n;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nheap_updates += nheap;

}




// Customized search_preassigned() for search_mode = 1, 2, 3.
// Added an input variable num_candidate_cluster because we do not use
// nprobe to determine the number of candidate clusters.
// We didn't implement OpenMP parallelization because we focused on
// single-thread performance in this work.
void IndexIVF::search_preassigned_custom (idx_t n, const float *x, idx_t k,
                                          const idx_t *keys,
                                          const float *coarse_dis,
                                          float *distances, idx_t *labels,
                                          bool store_pairs,
                                          long num_candidate_cluster,
                                          const IVFSearchParameters *params) const
{
    long nprobe = params ? params->nprobe : this->nprobe;

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    for (idx_t i = 0; i < n; i++) {
        InvertedListScanner *scanner = get_InvertedListScanner(store_pairs);
        ScopeDeleter1<InvertedListScanner> del(scanner);

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // intialize + reorder a result heap

        auto init_result = [&](float *simi, idx_t *idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP> (k, simi, idxi);
            } else {
                heap_heapify<HeapForL2> (k, simi, idxi);
            }
        };

        auto reorder_result = [&] (float *simi, idx_t *idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP> (k, simi, idxi);
            } else {
                heap_reorder<HeapForL2> (k, simi, idxi);
            }
        };

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&] (idx_t key, float coarse_dis_i,
                                    float *simi, idx_t *idxi) {

            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT (key < (idx_t) nlist,
                    "Invalid key=%ld nlist=%ld\n",
                    key, nlist);

            size_t list_size = invlists->list_size(key);

            // don't waste time on empty lists
            if (list_size == 0) {
                return (size_t)0;
            }

            scanner->set_list (key, coarse_dis_i);

            nlistv++;

            InvertedLists::ScopedCodes scodes (invlists, key);

            std::unique_ptr<InvertedLists::ScopedIds> sids;
            const Index::idx_t * ids = nullptr;

            if (!store_pairs)  {
                sids.reset (new InvertedLists::ScopedIds (invlists, key));
                ids = sids->get();
            }

            nheap += scanner->scan_codes (list_size, scodes.get(),
                                            ids, simi, idxi, k);

            return list_size;
        };

        /****************************************************
        * Actual loops, depending on search_mode
        ****************************************************/

        scanner->set_query (x + i * d);
        float * simi = distances + i * k;
        idx_t * idxi = labels + i * k;
        init_result (simi, idxi);
        long nscan = 0;

        if (search_mode == 1) { // generate training/testing data
            // 4 represents the number of intermediate search result features
            // at each pred_thresh timestamp.
            float * feature = new float [4 * pred_thresh.size()];
            float eps = 0.0000000001; // to avoid division by zero
            // Search clusters and record the intermediate search result
            // features at each pred_thresh timestamp.
            size_t start = 0;
            for (long j = 0; j < pred_thresh.size(); j++) {
                if (j != 0) {
                    start = pred_thresh[j-1];
                }
                for (size_t ik = start; ik < pred_thresh[j]; ik++) {
                    nscan += scan_one_list (
                        keys [i * num_candidate_cluster + ik],
                        coarse_dis[i * num_candidate_cluster + ik],
                        simi, idxi
                    );
                }
                // Reorder the heap before recording search result.
                reorder_result (simi, idxi);
                feature[j*4] = simi[0]; // top 1 intermediate search result
                feature[j*4+1] = simi[9]; // top 10 intermediate search result
                feature[j*4+2] = simi[0]/(simi[9]+eps); // top 1/top 10
                // Top 1/top 1 centroid-query distance.
                feature[j*4+3] = simi[0]/(coarse_dis[i*num_candidate_cluster]+eps);
                // Heapify the heap to continue the search.
                heap_heapify<HeapForL2> (k, simi, idxi, simi, idxi, k);
            }
            ndis += nscan;
            reorder_result (simi, idxi);
            // HACK: we overwrite the actual search result distances in simi
            // by the features so that we can easily write the features by
            // reading the search results without additional APIs.
            
            // Find the ground truth minimum termination condition in terms of
            // minimum number of nearest clusters to search. For IVF, gtvector
            // includes the cluster ids that have at least one of the ground
            // truth nearest neighbors. Note that we count the search as
            // successful as long as one of the ground truth nearest neighbors
            // is found (ties allowed)
            simi[0] = 0;
            for (long j = 0; j < num_candidate_cluster; j++) {
                for (int igt = 0; igt < gtvector[i].size(); igt++) {
                    if (keys[i*num_candidate_cluster+j] == gtvector[i][igt]) {
                        simi[0] = j+1;
                        break;
                    }
                }
                if (simi[0] != 0) {
                    break;
                }
            }
            // Distance(query, xth nearest cluster centroid) /
            // distance(query, 1st nearest cluster centroid)
            // where x = 10, 20, 30, ..., 90, 100.
            for (int j = 1; j < 11; j++) {
                simi[j] = coarse_dis[i*num_candidate_cluster+j*10-1]/
                    (coarse_dis[i*num_candidate_cluster]+eps);
            }
            // The recorded intermediate search results.
            for (int j = 0; j < 4*pred_thresh.size(); j++) {
                simi[11+j] = feature[j];
            }
            delete [] feature;
        } else if (search_mode == 2) { // learned early termination
            // term_cond is the termination condition computed as
            // min(max(prediction,1) * nprobe / 100.0, pred_max).
            // Here nprobe is used as a tunable multiplier.
            // For 1 billion database we actually predict the log of
            // termiantion condition so term_cond is computed as
            // min((2**max(prediction,0)) * nprobe / 100.0, pred_max).
            long term_cond = -1;
            float term_c = -1;
            int thresh_idx = 0; // current pred_thresh timestamp 
            double * input = new double[d+14]; // input features
            double * output = new double[1]; // prediction output
            double eps = 0.0000000001; // to avoid division by zero
            // Query vector features.
            for (idx_t j = 0; j < d; j++) {
                input[j] = (double)(x[i * d + j]);
            }
            // Distance(query, xth nearest cluster centroid) /
            // distance(query, 1st nearest cluster centroid)
            // where x = 10, 20, 30, ..., 90, 100.
            for (int j = 1; j < 11; j++) {
                input[d+j-1] = coarse_dis[i*num_candidate_cluster+j*10-1]/
                    (coarse_dis[i*num_candidate_cluster]+eps);
            }
            // Search up to top term_cond clusters. Whenever a pred_thresh is
            // reached make a prediction to update term_cond.
            for (size_t ik = 0; ik < pred_max; ik++) {
                nscan += scan_one_list (
                    keys [i * num_candidate_cluster + ik],
                    coarse_dis[i * num_candidate_cluster + ik],
                    simi, idxi
                );
                if (thresh_idx < pred_thresh.size() &&
                    ik+1 == pred_thresh[thresh_idx]) {
                    reorder_result (simi, idxi);
                    input[d+10] = simi[0]; // top 1 intermediate search result
                    input[d+11] = simi[9]; // top 10 intermediate search result
                    input[d+12] = simi[0]/(simi[9]+eps); // top 1/top 10
                    // Top 1/top 1 centroid-query distance.
                    input[d+13] = simi[0]/(coarse_dis[i*num_candidate_cluster]+eps);
                    // Make prediction.
                    (boosters[thresh_idx])->PredictRaw(input, output,
                        &tree_early_stop);
                    if (ntotal < 1000000000) {
                        term_cond = (long)(ceil(std::max((double)1,output[0])*
                            nprobe/100.0));
                        term_c = std::max((double)1,output[0]);
                    } else {
                        term_cond = (long)(ceil(pow(2.0,
                            std::max((double)0,output[0]))*nprobe/100.0));
                    }
                    heap_heapify<HeapForL2> (k, simi, idxi, simi, idxi, k);
                    thresh_idx++;
                }
                // Stop when termination condition reached.
                if (term_cond > 0 && ik+1 >= term_cond) {
                    printf("%.3f\n", term_c);
                    break;
                }
            }
            reorder_result (simi, idxi);
            ndis += nscan;
            delete [] input;
            delete [] output;
        } else if (search_mode == 3) { // simple heuristic approach
            size_t heur_nprobe = 0;
            // Use distance(query, 1st nearest cluster centroid)*nprobe/100.0
            // as threshold and search clusters with distance(query, centroid)
            // no more than the threshold.
            // Here nprobe is used as a tunable multiplier.
            float thresh = coarse_dis[i*num_candidate_cluster]
                *float(nprobe)/100.0;
            for (size_t j = 0; j < num_candidate_cluster; j++) {
                if (coarse_dis[i*num_candidate_cluster+j] <= thresh) {
                    heur_nprobe = j+1;
                } else {
                    break;
                }
            }
            for (size_t ik = 0; ik < heur_nprobe; ik++) {
                nscan += scan_one_list (
                    keys [i * num_candidate_cluster + ik],
                    coarse_dis[i * num_candidate_cluster + ik],
                    simi, idxi
                );
            }
            reorder_result (simi, idxi);
            ndis += nscan;
        }
    }
    indexIVF_stats.nq += n;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nheap_updates += nheap;
}

void IndexIVF::range_search (idx_t nx, const float *x, float radius,
                             RangeSearchResult *result) const
{
    std::unique_ptr<idx_t[]> keys (new idx_t[nx * nprobe]);
    std::unique_ptr<float []> coarse_dis (new float[nx * nprobe]);

    double t0 = getmillisecs();
    quantizer->search (nx, x, nprobe, coarse_dis.get (), keys.get ());
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists (keys.get(), nx * nprobe);

    range_search_preassigned (nx, x, radius, keys.get (), coarse_dis.get (),
                              result);

    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexIVF::range_search_preassigned (
         idx_t nx, const float *x, float radius,
         const idx_t *keys, const float *coarse_dis,
         RangeSearchResult *result) const
{

    size_t nlistv = 0, ndis = 0;
    bool store_pairs = false;

    std::vector<RangeSearchPartialResult *> all_pres (omp_get_max_threads());

#pragma omp parallel reduction(+: nlistv, ndis)
    {
        RangeSearchPartialResult pres(result);
        std::unique_ptr<InvertedListScanner> scanner
            (get_InvertedListScanner(store_pairs));
        FAISS_THROW_IF_NOT (scanner.get ());
        all_pres[omp_get_thread_num()] = &pres;

        // prepare the list scanning function

        auto scan_list_func = [&](size_t i, size_t ik, RangeQueryResult &qres) {

            idx_t key = keys[i * nprobe + ik];  /* select the list  */
            if (key < 0) return;
            FAISS_THROW_IF_NOT_FMT (
                  key < (idx_t) nlist,
                  "Invalid key=%ld  at ik=%ld nlist=%ld\n",
                  key, ik, nlist);
            const size_t list_size = invlists->list_size(key);

            if (list_size == 0) return;

            InvertedLists::ScopedCodes scodes (invlists, key);
            InvertedLists::ScopedIds ids (invlists, key);

            scanner->set_list (key, coarse_dis[i * nprobe + ik]);
            nlistv++;
            ndis += list_size;
            scanner->scan_codes_range (list_size, scodes.get(),
                                       ids.get(), radius, qres);
        };

        if (parallel_mode == 0) {

#pragma omp for
            for (size_t i = 0; i < nx; i++) {
                scanner->set_query (x + i * d);

                RangeQueryResult & qres = pres.new_result (i);

                for (size_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func (i, ik, qres);
                }

            }

        } else if (parallel_mode == 1) {

            for (size_t i = 0; i < nx; i++) {
                scanner->set_query (x + i * d);

                RangeQueryResult & qres = pres.new_result (i);

#pragma omp for schedule(dynamic)
                for (size_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func (i, ik, qres);
                }
            }
        } else if (parallel_mode == 2) {
            std::vector<RangeQueryResult *> all_qres (nx);
            RangeQueryResult *qres = nullptr;

#pragma omp for schedule(dynamic)
            for (size_t iik = 0; iik < nx * nprobe; iik++) {
                size_t i = iik / nprobe;
                size_t ik = iik % nprobe;
                if (qres == nullptr || qres->qno != i) {
                    FAISS_ASSERT (!qres || i > qres->qno);
                    qres = &pres.new_result (i);
                    scanner->set_query (x + i * d);
                }
                scan_list_func (i, ik, *qres);
            }
        } else {
            FAISS_THROW_FMT ("parallel_mode %d not supported\n", parallel_mode);
        }
        if (parallel_mode == 0) {
            pres.finalize ();
        } else {
#pragma omp barrier
#pragma omp single
            RangeSearchPartialResult::merge (all_pres, false);
#pragma omp barrier

        }
    }
    indexIVF_stats.nq += nx;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
}


InvertedListScanner *IndexIVF::get_InvertedListScanner (
    bool /*store_pairs*/) const
{
    return nullptr;
}

void IndexIVF::reconstruct (idx_t key, float* recons) const
{
    FAISS_THROW_IF_NOT_MSG (direct_map.size() == ntotal,
                            "direct map is not initialized");
    FAISS_THROW_IF_NOT_MSG (key >= 0 && key < direct_map.size(),
                            "invalid key");
    idx_t list_no = direct_map[key] >> 32;
    idx_t offset = direct_map[key] & 0xffffffff;
    reconstruct_from_offset (list_no, offset, recons);
}


void IndexIVF::reconstruct_n (idx_t i0, idx_t ni, float* recons) const
{
    FAISS_THROW_IF_NOT (ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size (list_no);
        ScopedIds idlist (invlists, list_no);

        for (idx_t offset = 0; offset < list_size; offset++) {
            idx_t id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }

            float* reconstructed = recons + (id - i0) * d;
            reconstruct_from_offset (list_no, offset, reconstructed);
        }
    }
}


void IndexIVF::search_and_reconstruct (idx_t n, const float *x, idx_t k,
                                       float *distances, idx_t *labels,
                                       float *recons) const
{
    idx_t * idx = new idx_t [n * nprobe];
    ScopeDeleter<idx_t> del (idx);
    float * coarse_dis = new float [n * nprobe];
    ScopeDeleter<float> del2 (coarse_dis);

    quantizer->search (n, x, nprobe, coarse_dis, idx);

    invlists->prefetch_lists (idx, n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned (n, x, k, idx, coarse_dis,
                        distances, labels, true /* store_pairs */);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = 0; j < k; ++j) {
            idx_t ij = i * k + j;
            idx_t key = labels[ij];
            float* reconstructed = recons + ij * d;
            if (key < 0) {
                // Fill with NaNs
                memset(reconstructed, -1, sizeof(*reconstructed) * d);
            } else {
                int list_no = key >> 32;
                int offset = key & 0xffffffff;

                // Update label to the actual id
                labels[ij] = invlists->get_single_id (list_no, offset);

                reconstruct_from_offset (list_no, offset, reconstructed);
            }
        }
    }
}

void IndexIVF::reconstruct_from_offset(
    idx_t /*list_no*/,
    idx_t /*offset*/,
    float* /*recons*/) const {
  FAISS_THROW_MSG ("reconstruct_from_offset not implemented");
}

void IndexIVF::reset ()
{
    direct_map.clear ();
    invlists->reset ();
    ntotal = 0;
}


Index::idx_t IndexIVF::remove_ids (const IDSelector & sel)
{
    FAISS_THROW_IF_NOT_MSG (!maintain_direct_map,
                    "direct map remove not implemented");

    std::vector<idx_t> toremove(nlist);

#pragma omp parallel for
    for (idx_t i = 0; i < nlist; i++) {
        idx_t l0 = invlists->list_size (i), l = l0, j = 0;
        ScopedIds idsi (invlists, i);
        while (j < l) {
            if (sel.is_member (idsi[j])) {
                l--;
                invlists->update_entry (
                     i, j,
                     invlists->get_single_id (i, l),
                     ScopedCodes (invlists, i, l).get());
            } else {
                j++;
            }
        }
        toremove[i] = l0 - l;
    }
    // this will not run well in parallel on ondisk because of possible shrinks
    idx_t nremove = 0;
    for (idx_t i = 0; i < nlist; i++) {
        if (toremove[i] > 0) {
            nremove += toremove[i];
            invlists->resize(
                i, invlists->list_size(i) - toremove[i]);
        }
    }
    ntotal -= nremove;
    return nremove;
}




void IndexIVF::train (idx_t n, const float *x)
{
    if (verbose)
        printf ("Training level-1 quantizer\n");

    train_q1 (n, x, verbose, metric_type);

    if (verbose)
        printf ("Training IVF residual\n");

    train_residual (n, x);
    is_trained = true;

}

void IndexIVF::train_residual(idx_t /*n*/, const float* /*x*/) {
  if (verbose)
    printf("IndexIVF: no residual training\n");
  // does nothing by default
}


void IndexIVF::check_compatible_for_merge (const IndexIVF &other) const
{
    // minimal sanity checks
    FAISS_THROW_IF_NOT (other.d == d);
    FAISS_THROW_IF_NOT (other.nlist == nlist);
    FAISS_THROW_IF_NOT (other.code_size == code_size);
    FAISS_THROW_IF_NOT_MSG (typeid (*this) == typeid (other),
                  "can only merge indexes of the same type");
}


void IndexIVF::merge_from (IndexIVF &other, idx_t add_id)
{
    check_compatible_for_merge (other);
    FAISS_THROW_IF_NOT_MSG ((!maintain_direct_map &&
                             !other.maintain_direct_map),
                  "direct map copy not implemented");

    invlists->merge_from (other.invlists, add_id);

    ntotal += other.ntotal;
    other.ntotal = 0;
}


void IndexIVF::replace_invlists (InvertedLists *il, bool own)
{
    //FAISS_THROW_IF_NOT (ntotal == 0);
    FAISS_THROW_IF_NOT (il->nlist == nlist &&
                        il->code_size == code_size);
    if (own_invlists) {
        delete invlists;
    }
    invlists = il;
    own_invlists = own;
}


void IndexIVF::copy_subset_to (IndexIVF & other, int subset_type,
                                 idx_t a1, idx_t a2) const
{

    FAISS_THROW_IF_NOT (nlist == other.nlist);
    FAISS_THROW_IF_NOT (code_size == other.code_size);
    FAISS_THROW_IF_NOT (!other.maintain_direct_map);
    FAISS_THROW_IF_NOT_FMT (
          subset_type == 0 || subset_type == 1 || subset_type == 2,
          "subset type %d not implemented", subset_type);

    size_t accu_n = 0;
    size_t accu_a1 = 0;
    size_t accu_a2 = 0;

    InvertedLists *oivf = other.invlists;

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t n = invlists->list_size (list_no);
        ScopedIds ids_in (invlists, list_no);

        if (subset_type == 0) {
            for (idx_t i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (a1 <= id && id < a2) {
                    oivf->add_entry (list_no,
                                     invlists->get_single_id (list_no, i),
                                     ScopedCodes (invlists, list_no, i).get());
                    other.ntotal++;
                }
            }
        } else if (subset_type == 1) {
            for (idx_t i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (id % a1 == a2) {
                    oivf->add_entry (list_no,
                                     invlists->get_single_id (list_no, i),
                                     ScopedCodes (invlists, list_no, i).get());
                    other.ntotal++;
                }
            }
        } else if (subset_type == 2) {
            // see what is allocated to a1 and to a2
            size_t next_accu_n = accu_n + n;
            size_t next_accu_a1 = next_accu_n * a1 / ntotal;
            size_t i1 = next_accu_a1 - accu_a1;
            size_t next_accu_a2 = next_accu_n * a2 / ntotal;
            size_t i2 = next_accu_a2 - accu_a2;

            for (idx_t i = i1; i < i2; i++) {
                oivf->add_entry (list_no,
                                 invlists->get_single_id (list_no, i),
                                 ScopedCodes (invlists, list_no, i).get());
            }

            other.ntotal += i2 - i1;
            accu_a1 = next_accu_a1;
            accu_a2 = next_accu_a2;
        }
        accu_n += n;
    }
    FAISS_ASSERT(accu_n == ntotal);

}

// Load cluster indexes where the ground truth nearest neighbor(s) reside.
// This is for finding ground truth minimum termination condition.
void IndexIVF::load_gt(long label)
{
    if (label == -1){
        gtvector.clear();
    } else if (label == -2){
        std::vector<idx_t> newquery;
        gtvector.push_back(newquery);
    } else {
        gtvector[gtvector.size()-1].push_back(label);
    }
}

// Load the thresholds about when to make predictions.
// This is related to the choice of intermediate search result features.
void IndexIVF::load_thresh(long thresh)
{
    if (thresh == -1){
        pred_thresh.clear();
    } else {
        pred_thresh.push_back(thresh);
    }
}

// Load the prediction model.
void IndexIVF::load_model(char *file)
{
    std::string filename(file);
    LightGBM::Boosting *booster =
        LightGBM::Boosting::CreateBoosting(std::string("gbdt"),
        filename.c_str());
    boosters.push_back(booster);
}

IndexIVF::~IndexIVF()
{
    if (own_invlists) {
        delete invlists;
    }
}


void IndexIVFStats::reset()
{
    memset ((void*)this, 0, sizeof (*this));
}


IndexIVFStats indexIVF_stats;

void InvertedListScanner::scan_codes_range (size_t ,
                       const uint8_t *,
                       const idx_t *,
                       float ,
                       RangeQueryResult &) const
{
    FAISS_THROW_MSG ("scan_codes_range not implemented");
}



} // namespace faiss
