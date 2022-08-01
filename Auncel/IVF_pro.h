/**
 * Copyright (c) Zili Zhang
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef PROFILE_FOR_IVF
#define PROFILE_FOR_IVF

#include "utils.h"
#include "Index.h"
#include "IndexBinary.h"

#include <vector>
#include <string> 
#include <utility>
#include <sys/time.h>

namespace faiss {

using idx = Index::idx_t;

// calculate the L2 dis between two arbitrary vectors in a vector set
void fvec_inter_vecs(float * ret, const float * x, size_t num, size_t d);

// calculate the IP dis between two arbitrary vectors in a vector set
void fvec_inter_vecs_IP(float * ret, const float * x, size_t num, size_t d);

/// calculate the dis between query vector and boundary
// a < b
float cosine_theorem(float a, float b, float c);

/// apply binary search method  to retrieve val, return its index
// size_t binary_search(std::vector<float> &vec, float val);


/// calculate the index scaling val
float kscaling(float kdis, size_t in, const float *gt_id, size_t max_topk);

// Map: Sum of angles -> k's scaling up
struct Trace{
    // Trace for nprobe
    size_t nprobe;

    // Map attributes
    std::vector<std::pair<float, float> > trace;
    
    // Standard deviation of values
    std::vector<float> stds;

    size_t bs = 250;

    // Binary search
    float search(float k, float std_m);

    // Sort and Batch
    void SB();

};

using Traces = std::vector<Trace>;

struct TrainPoint{
    std::vector<float> acc;

    std::string key;

    size_t key_value;

    std::vector<float> topk_dis;

    std::vector<idx> topk_id;

};

class error_pro{

public:
    float std_m = 1.0;

    float multipler = 1.0;

    size_t arcos_size = 500;

    std::vector<float> arcos_list;

    /*Online part*/
    const float *require_acc;

    bool profile = false;

    bool overhead_profile = false;

    bool time_tune = false;

    size_t alloc_s;

    float* KD = nullptr;

    float* t_recalls = nullptr;


    float cur_rc;

    size_t *my_nprobe  = nullptr;

    size_t id;

    size_t count;

    size_t query_topk = -1;

    float *query_vec;

    /// Distance between cluster centers in current probe order
    // std::vector<float> cenTocen;

    // std::vector<float> disToboundary;

    /*Offline part*/
    std::vector<Trace> traces;

    enum metric{
        L2,
        IP
    };
    metric m_type;

    size_t nlist;       //num of clusters

    size_t max_topk;

    size_t d;       //Dimension

    size_t train_num;   //Size of the training set 

    float *interdis_cem;    //Distance between cluster centers(size: (nlist - 1)*nlist/2 )

    const float *train_q;   //query vectors in training dataset (size: train_num * d)

    const float *train_D;   //query vectors ground truth returned distance (size: train_num * max_topk)

    const idx *train_I;

    float *train_cd;  //train_q 's distance to nlist centroids (size: train_num * nlist )

    idx *train_ci;

    std::vector<TrainPoint> tps;    //Different Error searching trace for IVF calibrating

    float arcos(float x);

    void construct_arcos();

    // Geometric description

    void train(MetricType metric_type);

    void set_online(size_t _id, size_t topk, const float *cd, 
        const long* ci, const float *interdis_cem, std::vector<float> &disToboundary,
        std::vector<float> &cenTocen, bool training= false);

    size_t cur_num(const float*D, int id, std::vector<float> &disToboundary, size_t index, size_t query_k);

    /// calculate the sum of the angle for a query's k's nn when finishing searching in nprobe clusters
    float sum_angle(float kdis, float *disToboundary, size_t n, size_t start = 1);

    void setparam(int id);

    double time() {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        return tv.tv_sec + tv.tv_usec * 1e-6;
    }

    ~error_pro();
};

}
#endif