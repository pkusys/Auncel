// Copyright (c) Zili Zhang.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

#include "../IndexIVF.h"
#include "../IndexFlat.h"
#include "../profile.h"
#include "../AutoTune.h"
#include <omp.h>

#include<iostream>
#include<sstream>
#include<fstream>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/
double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, sizeof(int), 1, f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d,n;
    fread(&n, sizeof(int), 1, f);
    fread(&d, sizeof(int), 1, f);
    printf("d : %d, n: %d\n", d, n);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    *d_out = d;
    *n_out = n;
    int64_t total_size = int64_t(d) * int64_t(n);
    int64_t slice_size = total_size/20;
    float* x = new float[total_size];
    int64_t nr = 0;
    double t0 = elapsed();
    for (int i = 0; i<20; i++){
        nr += fread(x, sizeof(float), slice_size, f);
        x+=slice_size;
        printf("Read %d/100 slice done...(%.3fs)\n", i, elapsed()-t0);
        t0 = elapsed();
    }
    // int64_t nr = fread(x, sizeof(float), total_size, f);
    printf("Read finished, read %lld\n", nr);
    assert(nr == total_size || !"could not read whole file");
    fclose(f);
    return x - (total_size);
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}


void fvecs_write(const char* fname, float* data ,size_t* d_in, size_t* n_in) {
    FILE* f = fopen(fname, "wb");
    size_t d = *d_in;
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    for(size_t i =0;i < *n_in;i++){
        fwrite(d_in, sizeof(int), 1, f);
        fwrite(data + (i * d), sizeof(float), d, f);
    }

    fclose(f);
}

int main(int argc,char **argv) {
    if(argc - 1 != 1){
        printf("You at least need one param(s): dbfile.\n");
        return 0;
    }
    double t0 = elapsed();
    std::string dbfile = argv[1];
    int dbfilename_s = dbfile.size();
    std::string gtdis = dbfile.substr(0, dbfilename_s - 7) + "gtd.fvecs";
    std::string gtidx = dbfile.substr(0, dbfilename_s - 7) + "gti.ivecs";
    std::cout << gtdis << std::endl << gtidx <<std::endl;

    const char* index_key = "Flat";
    // const char *index_key = "HNSW32,Flat";

    faiss::Index* index;
    size_t d;
    size_t topk = 100;

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float* xt = fvecs_read("/workspace/data/dist/deep10M.fvecs", &d, &nt);
        // float* x = fvecs_read("../../../deep10M_data/10M_querys.fvecs", &dd, &nn);


        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0,
               index_key,
               d);
        index = new faiss::IndexFlatL2(d);


        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->train(nt, xt);
        printf("[%.3f s] Training finished\n", elapsed() - t0);
        delete[] xt;
    }
    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        float* xb = fbin_read(dbfile.c_str(), &d2, &nb);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               nb,
               d);

        index->add(nb, xb);

        delete[] xb;
    }
    size_t nq;
    float* xq;

    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("/workspace/data/dist/deep1B_queries.fvecs", &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }
    /*This aimed to get groud truth for sample queries*/
    nq = 10000;
    std::vector<faiss::Index::idx_t> I(nq * topk);
    std::vector<float> D(nq * topk);

    omp_set_num_threads(20);
    printf("[%.3f s] Start searching\n", elapsed() - t0);
#pragma omp parallel for
    for(int i = 0; i<20;i++){
        index->search(10, xq + i*10*index->d, topk, D.data() + i*10*topk, I.data()+ i*10*topk);
        printf("[%.3f s] in searching\n", elapsed() - t0);
    }
    printf("[%.3f s] search finished\n", elapsed() - t0);

    /*save gt values into files*/
    nq = 200;
    std::vector<int> int_I(nq * topk);
    for(int j = 0;j<nq*topk;j++)
        int_I[j] = I[j];
    FILE* f = fopen(gtidx.c_str(), "wb");
    for(size_t i = 0; i < nq; i++){
        fwrite(&topk, sizeof(int), 1, f);
        fwrite(int_I.data() + (topk * i), sizeof(int), topk, f);
    }
    fclose(f);

    FILE* tf = fopen(gtdis.c_str(), "wb");
    for(size_t i = 0; i < nq; i++){
        fwrite(&topk, sizeof(int), 1, tf);
        fwrite(D.data() + (topk * i), sizeof(float), topk, tf);
    }
    fclose(tf);
    delete[] xq;
    delete index;
    return 0;
}
