/**
 * Copyright (c) Zili Zhang
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "profile.h"
#include "FaissAssert.h"
#include "utils.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <typeinfo>
#include <fstream>
#include <iostream>
#include <sstream>

// non-const version
#define DC(classname) classname* ix = dynamic_cast<classname*>(index)
#define DCX(classname) classname* ix = dynamic_cast<classname*>(in)

namespace faiss {

Error_sys::Error_sys( Index *in, size_t nq, size_t topk): 
        train_num(nq), max_topk(topk) {
    FAISS_THROW_IF_NOT_MSG(
        nq%10 == 0, "Train num must be evenly divided by ten");
    index = nullptr;
    key = "Base";
    if(DCX( IndexIVF)){
        index = ix;
        key = "IVF";
        ix->t = nullptr;
    }
}

Error_sys::Error_sys(){
    /// Do nothing
}

void Error_sys::set_gt(const float* gt_D_in, const  Index::idx_t* gt_I_in){
    FAISS_THROW_IF_NOT_MSG((
        gt_D_in != nullptr && 
        gt_I_in != nullptr), 
        "the ground truth must not be null ptr when setting up");
    train_D.resize(train_num * max_topk);
    memcpy(train_D.data(), gt_D_in, sizeof(train_D[0]) * train_num * max_topk);
    train_I.resize(train_num * max_topk);
    memcpy(train_I.data(), gt_I_in, sizeof(train_I[0]) * train_num * max_topk);
}

void Error_sys::set_train_point(float *D,  Index::idx_t*I, size_t key_v, size_t nq){
    FAISS_THROW_IF_NOT_MSG((index->t != nullptr), 
        "your must init tune for index first");
    FAISS_THROW_IF_NOT_MSG(
            (train_I.size() == train_num * max_topk),
            "ground truth not initialized");
    TrainPoint tmp;

    if (key == "IVF"){
        tmp.key = "nprobe";
        tmp.key_value = key_v;
    }

    tmp.topk_dis.resize(nq * max_topk);
    tmp.topk_id.resize(nq * max_topk);
    tmp.acc.resize(nq);
    memcpy(tmp.topk_dis.data(), D, sizeof(tmp.topk_dis[0]) * nq * max_topk);
    memcpy(tmp.topk_id.data(), I, sizeof(tmp.topk_id[0]) * nq * max_topk);
// #pragma omp parallel for
    for (size_t i = 0; i < nq;i++){
        // std::cout<<"Setted train point " << i <<std::endl;
        float tmp_res = recall(I + i * max_topk, 
            train_I.data() + i * max_topk, max_topk);
        // size_t tmp_res =  ranklist_intersection_size(
        //     max_topk, train_I.data() + i * max_topk, max_topk, I + i * max_topk);
        tmp.acc[i] = tmp_res;
    }
    if(DC( IndexIVF)){
        ix->t->tps.push_back(tmp);
    }
}

void Error_sys::sys_train(size_t nq, const float *xq){
    FAISS_THROW_IF_NOT_MSG(
            nq <= this->train_num, "Error sys training does not have the same nb of queries compared with creation");
    FAISS_THROW_IF_NOT_MSG(
            (train_I.size() == train_num * max_topk),
            "ground truth not initialized");
    // IVF index
    if(DC( IndexIVF)){
        // init IVF tuner(hack) method
        ix->init_tune(nq, max_topk, xq, train_D.data(), train_I.data(), nullptr, nullptr);
        size_t nlist = ix->nlist;
        std::cout << "Init IVF done" << std::endl;
        ix->set_train_mode();
        // Just train nprobe=1,2,4...128 case for init
        size_t nprobe = nlist, max_np = nlist;
        for (; nprobe<=max_np; nprobe <<= 1){
            ix->nprobe = nprobe;
            std::vector<float> D(nq * max_topk);
            std::vector< Index::idx_t> I(nq * max_topk);
            std::vector<float> center_dis(nq * nlist);
            std::vector< Index::idx_t> center_id(nq * nlist);
            size_t batchsize = nq/10;
            double t0 = time();
// #pragma omp parallel for
            for (size_t q0 = 0; q0 < nq; q0 += batchsize) {
                size_t q1 = q0 + batchsize;
                if (q1 > nq)
                    q1 = nq;
                // if (ix->t->train_cd == nullptr){
                //     ix->search(
                //             q1 - q0,
                //             xq + q0 * ix->d,
                //             max_topk,
                //             D.data() + q0 * max_topk,
                //             I.data() + q0 * max_topk,
                //             center_dis.data() + q0 * nlist,
                //             center_id.data() + q0 * nlist,
                //             q0);
                // }
                // else{
                    ix->search(
                            q1 - q0,
                            xq + q0 * ix->d,
                            max_topk,
                            D.data() + q0 * max_topk,
                            I.data() + q0 * max_topk,
                            q0);
                // }
            }

            // if (ix->t->train_cd == nullptr){
            //     ix->t->train_cd = new float[nq * nlist];
            //     ix->t->train_ci = new  Index::idx_t[nq * nlist];
            //     memcpy(ix->t->train_cd, center_dis.data(), sizeof(center_dis[0])*nq * nlist);
            //     memcpy(ix->t->train_ci, center_id.data(), sizeof(center_id[0])*nq * nlist);
            // }
            double t1 = time();
            printf("System search time: %.3f\n", t1-t0);
            // std::cout<<"Setting train point" <<std::endl;
            set_train_point(D.data(), I.data(), nprobe, nq);
            // std::cout<<"Setted train point" <<std::endl;
        }
        ix->set_train_off();
        this->is_trained = true;

        std::cout<<"Start t traing" <<std::endl; // Sort and compress maps
        ix->t->train( METRIC_L2);       
        std::cout<<"End t traing" <<std::endl;

        /*test offline part*/
        for (size_t ij = 0; ij < 8; ij++){
            size_t np = (1 << ij);
            std::stringstream ss;
            ss<<"Validation_"<< ix->d << "_" << np <<".log";
            std::string filename = ss.str();

            std::ofstream outfile;
            outfile.open(filename);
            for(int i = 0;i < ix->t->traces[ij].trace.size(); i++){
                outfile << ix->t->traces[ij].trace[i].first << " " << ix->t->traces[ij].trace[i].second << std::endl;
            }
        }
    }
}

void Error_sys::set_queries(size_t n, const float *q, const float*acc, size_t allo_size){
    num = n;
    queries = q;
    require_acc = acc;
    if (DC( IndexIVF)){
        ix->t->alloc_s = allo_size;
        if (ix->t->my_nprobe != nullptr)
            delete[] ix->t->my_nprobe;
        ix->t->my_nprobe = new size_t[allo_size];
        memset(ix->t->my_nprobe, 0, allo_size*sizeof(ix->t->my_nprobe[0]));

        size_t ind = 0;
        size_t nprobe = 1;
        while(nprobe <= ix->nlist/8){
            nprobe <<= 1;
            ind++;
        }
        if (ix->t->KD != nullptr)
            delete[] ix->t->KD;
        ix->t->KD = new float[allo_size*ind];
        memset(ix->t->KD, 0, ind*allo_size*sizeof(ix->t->KD[0]));

        if (ix->t->t_recalls != nullptr)
            delete[] ix->t->t_recalls;
        ix->t->t_recalls = new float[allo_size];
        memset(ix->t->t_recalls, 0, allo_size*sizeof(ix->t->t_recalls[0]));

        ix->t->require_acc = acc;
    }
}

void Error_sys::set_topk(size_t new_topk){
    if(key == "IVF"){
        DC( IndexIVF);
        ix->t->query_topk = new_topk;
    }
}

void Error_sys::search(float *D, int64_t *I, size_t start, size_t search_size){
    FAISS_THROW_IF_NOT_MSG(
            is_trained == true, "Error sys must be trained before searching");
    FAISS_THROW_IF_NOT_MSG(
            num <= train_num, "Error sys search num must be lower than all qeuries num");
    index->set_tune_mode();
    if(key == "IVF"){
        DC( IndexIVF);
        ix->set_tune_mode(); 
        ix->nprobe = ix->nlist;
        if (search_size == -1)
            ix->search(num, queries +start * ix->d, max_topk, D, I, start);
        else
            ix->search(search_size, queries +start * ix->d, max_topk, D, I, start);
    }
    index->set_tune_off();
}

void Error_sys::time_search(float *D, int64_t *I, size_t start, size_t search_size){
    FAISS_THROW_IF_NOT_MSG(
            is_trained == true, "Error sys must be trained before searching");
    FAISS_THROW_IF_NOT_MSG(
            num <= train_num, "Error sys search num must be lower than all qeuries num");
    if(key == "IVF"){
        DC( IndexIVF);
        ix->t->time_tune = true;
        ix->nprobe = ix->nlist;
        if (search_size == -1)
            ix->search(num, queries +start * ix->d, max_topk, D, I, start);
        else
            ix->search(search_size, queries +start * ix->d, max_topk, D, I, start);
        ix->t->time_tune = true;
    }
}

float Error_sys::recall( Index::idx_t *I,  Index::idx_t *gtI, size_t topk){
    idx* v2 = I;
    std::sort(v2, v2 + topk);
    { // de-dup v2
        int64_t prev = -1;
        size_t wp = 0;
        for (size_t i = 0; i < topk; i++) {
            if (v2[i] != prev) {
                v2[wp++] = prev = v2[i];
            }
        }
        topk = wp;
    }
    const int64_t seen_flag = int64_t{1} << 60;
    size_t count = 0;
    for (size_t i = 0; i < topk; i++) {
        int64_t q = gtI[i];
        size_t i0 = 0, i1 = topk;
        while (i0 + 1 < i1) {
            size_t imed = (i1 + i0) / 2;
            int64_t piv = v2[imed] & ~seen_flag;
            if (piv <= q)
                i0 = imed;
            else
                i1 = imed;
        }
        if (v2[i0] == q) {
            count++;
            v2[i0] |= seen_flag;
        }
    }
    // delete[] v2;

    return float(count)/topk;
}

}