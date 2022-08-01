/**
 * Copyright (c) Zili Zhang
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef ERROR_PROFILE_SYS
#define ERROR_PROFILE_SYS

#include "Index.h"
#include "IndexBinary.h"
#include "IndexIVF.h"
#include "IVF_pro.h"

#include <condition_variable>
#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>

namespace faiss{

class Error_sys {
public:
    /// Pending queries
    const float *queries;

    size_t num;

    const float *require_acc;

    bool is_trained = false;

    /// Train attributes
    std::string key;

    size_t train_num;

    size_t max_topk;

    IndexIVF *index;

    std::vector<float> train_D;   //query vectors ground truth returned distance (size: train_num * max_topk)

    std::vector<Index::idx_t> train_I;

    /// init for error profile system
    Error_sys(Index *in, size_t nq, size_t topk);

    Error_sys();

    /// Set up the ground truth values for pre-training between building index and online query
    void set_gt(const float* gt_D_in, const  Index::idx_t* gt_I_in);

    /// Set up every different search for different nrpobes
    void set_train_point(float *D,  Index::idx_t*I, size_t key_v, size_t nq);

    /* Start pre-training between building index and online query, 
    mainly constains some search and init the sepecific hacking method 
    and algorithm for index searching  */
    void sys_train(size_t nq, const float *xq);

    /// Set online queries part
    void set_queries(size_t n, const float *q, const float*acc, size_t allo_size);

    /// Do error-profile sys search
    void search(float *D, int64_t*I, size_t start, size_t search_size = -1);

    void time_search(float *D, int64_t*I, size_t start, size_t search_size = -1);

    void set_topk(size_t new_topk);

    double time() {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }



    /// Testing sys performance
    float recall( Index::idx_t *I,  Index::idx_t *gtI, size_t topk);


};



}
#endif