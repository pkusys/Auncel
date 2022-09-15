/**
 * Copyright (c) Zili Zhang
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "FaissAssert.h"
#include "IVF_pro.h"
#include <algorithm>
#include <math.h>

#include <stdio.h>
#include <iostream>
#include <fstream>

namespace faiss {

void fvec_inter_vecs(float * ret, const float * x, size_t num, size_t d){
#pragma omp parallel for
    for (int64_t i = 0; i < num; i++){
        for (int64_t j = i + 1; j < num; j++){
            ret[(2*num - 1 -i)*i/2 + j -1 - i] =  fvec_L2sqr(x + i * d, x + j * d, d);
            // printf("res %f\n", ret[(2*num - 2 -i)*i/2 + j -1]);
        }
    }
}

void fvec_inter_vecs_IP(float * ret, const float * x, size_t num, size_t d){
#pragma omp parallel for
    for (int64_t i = 0; i < num; i++){
        for (int64_t j = i + 1; j < num; j++){
            ret[(2*num - 1 -i)*i/2 + j -1 - i] =  fvec_inner_product(x + i * d, x + j * d, d);
            // printf("res %f\n", ret[(2*num - 2 -i)*i/2 + j -1]);
        }
    }
}

float cosine_theorem(float a, float b, float c){
    FAISS_THROW_IF_NOT_MSG((a <= b), 
            "cosine theorem's prerequisites");
    // float temp = a + c - b;
    // float sqrt_c = std::sqrt(c);
    // temp = temp / (2 * sqrt_c);
    // return std::pow(sqrt_c/2 - temp, 2);
    float temp = pow(a, 2) + pow(c, 2) - pow(b, 2);
    temp = temp / (2 * c);
    return c/2 - temp;
}

// size_t binary_search(std::vector<float> &vec, float val){
//     size_t high = vec.size() - 1, low = 0;
//     size_t middle = 0;
//     while(low <= high) {
// 		middle = (low + high)/2;
//         if(vec[middle] >= val){
//             high = middle - 1;
//         }
//         else{
//             low = middle + 1;
//         }
//         // printf("middle: %d high: %d low: %d  %f\n", middle, high, low,vec[middle]);
// 	}
//     if (vec[low] > val)
//         low--;
//     return low + 1;
// }


float kscaling(float kdis, size_t in, const float *gt_id, size_t max_topk){
    size_t index = 0;
    // std::cout << gt_id[index] << std::endl;
    for( ;index < max_topk; index++){
        if( fabs(gt_id[index] - kdis)/kdis < 1e-5 || fabs(gt_id[index] - kdis) < 1e-5)
            break;
    }
    if(index >= max_topk)
        return -1;
    return (index + 1)/float(in + 1);
}

float Trace::search(float k, float std_m){
    float sc = std_m;
    if (k <= trace[0].first)
        return trace[0].second + sc*stds[0];
    // Dealing with upward cross-border situations
    if (k >= trace[trace.size() - 1].first){
        float ampli = k/trace[trace.size() - 1].first;
        return (trace[trace.size() - 1].second + sc*stds[trace.size() - 1])*ampli;
    }
    size_t high = trace.size() - 1, low = 0;
    size_t middle = 0;
    while(low <= high) {
		middle = (low + high)/2;
        if(trace[middle].first < k){
            low = middle + 1;
        }
        else{
            high = middle - 1;
        }
	}
    if (trace[low].first > k)
        low--;
    return trace[low].second + sc*stds[low];
}

void Trace::SB(){
    std::sort(trace.begin(), trace.end(), [](std::pair<float,float> &left, std::pair<float,float> &right) {
        return left.first > right.first;});
    
    size_t size = 0;
    std::for_each (std::begin(trace), std::end(trace), [&](const std::pair<float,float> d) {
        size += (d.first < 0 && d.second < 0)?0:1;
    });
    size_t sz = (size + bs - 1)/bs;
    std::vector<std::pair<float, float> > tmp_trace;
    tmp_trace.resize(sz);
    stds.resize(sz);

    for(size_t i = 0; i<sz; i++){
        size_t left = i*bs;
        size_t right = (i+1)*bs;
        right = std::min(right, size);

        float ave1 = 0, ave2 = 0, std = 0;

        // Prevent overflow in float, may reduce peformance
        for (size_t index = left; index < right; index++){
            size_t j = index - left;
            ave1 = (float)j/(float)(j+1) * ave1 + trace[index].first/(j+1);
            ave2 = (float)j/(float)(j+1) * ave2 + trace[index].second/(j+1);
        }
        double accum  = 0.;
        // Std won't overflow for most time
	    std::for_each (std::begin(trace) + left, std::begin(trace) + right, [&](const std::pair<float,float> d) {
		    accum  += (d.second-ave2)*(d.second-ave2);
        });
        std = std::sqrt(accum/bs);
        tmp_trace[i] = std::make_pair(ave1, ave2);
        stds[i] = std;
    }
    trace.resize(sz);
    memcpy(trace.data(), tmp_trace.data(), sz*sizeof(std::pair<float, float>));
    // Ascending order
    std::reverse(trace.begin(), trace.end());
    std::reverse(stds.begin(), stds.end());
}

void error_pro::construct_arcos(){
    int len = arcos_size;
    arcos_list.resize(len);
    float sc = len/2;
    for(int i = 0; i < len; i++){
        float x = float(i - sc) / sc;
        float y = std::acos(x);
        arcos_list[i] = y;
    }
}

float error_pro::sum_angle(float kdis, float *disToBoundary, size_t n, size_t start){
    float sum = 0;
    size_t end = start + n;
// #pragma omp simd
    for(size_t i = start; i < end; i++){
        if(disToBoundary[i] >= kdis)
            continue;
        // To reduce overhead, maybe we don't need to apply acos method
        // float angle = disToBoundary[i]/kdis;
        float angle = arcos(disToBoundary[i]/kdis);
        // angle = std::sin(angle);
        // angle = std::pow(angle, 16);
        sum += angle;
    }
    return sum;
}

float error_pro::arcos(float x){
    FAISS_THROW_IF_NOT_MSG((x <= 1. && x >= -1.), 
            "arcos's domain definition is [-1, 1]");
    int index = x*arcos_size/2 + arcos_size/2;
    return arcos_list[index];
}

void error_pro::train( MetricType metric_type){
    size_t i = 0;
    while((1<<i) <= nlist/8){
        std::cout<<"SB() "<< (1<<i) <<std::endl;
        // Sort and batch the map
        traces[i].SB();
        i++;
    }
}

void error_pro::set_online(size_t _id, size_t _topk, const float *_cd, 
        const idx* _ci, const float *interdis_cem, std::vector<float> &disToBoundary,
        std::vector<float> &cenTocen, bool training){
    id = _id;
    if (!training)
        cur_rc = require_acc[id];
    size_t max_num = nlist/8 + 20; // You should amplify 20 if you intend to collect more angles's sum.
    size_t topk = _topk;
    size_t cur_cen = _ci[0];
    cenTocen.resize(max_num);
    disToBoundary.resize(max_num);
    std::vector<float> cend(max_num);
    if(m_type == IP){
        for(int i = 0; i < max_num; i++)
            cend[i] = arcos(_cd[i]);
    }
    size_t i,j;
// #pragma omp parallel for
    for (size_t k = 1; k <= max_num; k++){
        size_t dstcen = _ci[k];
        i = cur_cen < dstcen? cur_cen : dstcen;
        j = cur_cen < dstcen? dstcen: cur_cen;
        float tmp = interdis_cem[(2*nlist - 1 -i)*i/2 + j -1 - i];
        cenTocen[k - 1] = tmp;
        // std::cout << "Testing point"  << " " << k << std::endl;
    }
// #pragma omp parallel for
    // std::ofstream outfile;
    // outfile.open("disBoundary.txt", std::ios::out|std::ios::app);
    for (size_t k = 0; k < max_num - 1; k++){
        // printf("%f, %f\n",_cd[0], _cd[k + 1]);
        float tmp;
        if(m_type == L2)
            tmp = cosine_theorem(_cd[0], _cd[k + 1], cenTocen[k]);
        else
            tmp = cosine_theorem(cend[0], cend[k + 1], cenTocen[k]);
        disToBoundary[k] = tmp;
        // printf("Dis boundary: %f\n", tmp);
        // outfile << tmp << " ";
    }
    // outfile << std::endl;
    // outfile.close();
}

void error_pro::setparam(int id){
    std::string fn = "../hyperparameter.txt";
    std::ifstream infile;
    infile.open(fn);

    for (int i = 0; i < 12; i++){
        float a, b;
        infile >> a >> b;
        if (i == id - 1){
            multipler = a;
            std_m = b;
            // std::cout << a << " " << b << "\n";
        }
    }

    profile = false;
}

size_t error_pro::cur_num(const float*D, int id, std::vector<float> &disToBoundary, size_t index, size_t query_k){
    std::vector<float> index_k(max_topk);
    size_t nprobe = 1 << index;
    // for(size_t i = 0; i < max_topk; i++){
    //     float sum_a;
    //     sum_a = sum_angle(D[i], disToBoundary.data(), 15, nprobe - 1);
    //     float k_scaling = traces[index].search(sum_a);
    //     index_k[i] = (i + 1) * k_scaling;
    // }
    // // printf("Time Cost: %f %f\n", t0, t1);
    // size_t num = 0;
    // for(size_t i = 0; i<max_topk; i++){
    //     if (index_k[i] <= query_k)
    //         num++;
    // }
    // return num;
    size_t high = query_k - 1, low = 0;
    size_t middle = 0;
    if (query_k*traces[index].search(sum_angle(D[high], disToBoundary.data(), 15, nprobe - 1), std_m) <= query_k*1.005)
        return query_k;
    while(low <= high) {
		middle = (low + high)/2;
        if (middle <= 0)
            return 0;
        if((middle+1)*traces[index].search(sum_angle(D[middle], disToBoundary.data(), 15, nprobe - 1), std_m) <= query_k){
            low = middle + 1;
        }
        else{
            high = middle - 1;
        }
	}
    return low+1;

}

error_pro::~error_pro(){
    delete[] train_ci;
    delete[] train_cd;
    delete[] my_nprobe;
    delete[] KD;
}

}
