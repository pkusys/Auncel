#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>
#include <omp.h>

#include "../AutoTune.h"
#include "../IndexIVF.h"
#include "../index_io.h"
#include "../profile.h"


#include<iostream>
#include<fstream>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

#define DC(classname) classname* ix = dynamic_cast<classname*>(index)

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

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/// Command like this: ./knn_script sift1M 100 2000 8000
int main(int argc,char **argv) {
    std::cout << argc << " arguments" <<std::endl;
    if(argc - 1 != 4){
        printf("You should at least input 4 params: the dataset name, topk, train size, query size\n");
        return 0;
    }
    std::string param1 = argv[1];
    std::string param2 = argv[2];
    std::string p3 = argv[3];
    std::string p4 = argv[4];

    int input_k = std::stoi(param2);
    int ts = std::stoi(p3);
    int ses = std::stoi(p4);

    if(input_k>100 || input_k <0){
        printf("Input topk must be lower than or equal to 100 and greater than 0\n");
        return 0;
    }
    std::string db, query, gtI, gtD;
    if(param1 == "sift1M"){
        db = "/workspace/data/sift/sift1M.fvecs";
        query = "/workspace/data/sift/1M_query.fvecs";
        gtI = "/workspace/data/sift/idx_1M.ivecs";
        gtD = "/workspace/data/sift/dis_1M.fvecs";
    }
    else if(param1 == "sift10M"){
        db = "/workspace/data/sift/sift10M/sift10M.fvecs";
        query = "/workspace/data/sift/sift10M/query.fvecs";
        gtI = "/workspace/data/sift/sift10M/idx.ivecs";
        gtD = "/workspace/data/sift/sift10M/dis.fvecs";
    }
    else if(param1 == "deep10M"){
        db = "/workspace/data/deep/deep10M.fvecs";
        query = "/workspace/data/deep/query.fvecs";
        gtI = "/workspace/data/deep/idx.ivecs";
        gtD = "/workspace/data/deep/dis.fvecs";
    }
    else if(param1 == "gist"){
        db = "/workspace/data/gist/gist1M.fvecs";
        query = "/workspace/data/gist/query.fvecs";
        gtI = "/workspace/data/gist/idx.ivecs";
        gtD = "/workspace/data/gist/dis.fvecs";
    }
    else if(param1 == "spacev"){
        db = "/workspace/data/spacev/spacev10M.fvecs";
        query = "/workspace/data/spacev/query.fvecs";
        gtI = "/workspace/data/spacev/idx.ivecs";
        gtD = "/workspace/data/spacev/dis.fvecs";
    }
    else if(param1 == "glove"){
        db = "/workspace/data/glove/glove.fvecs";
        query = "/workspace/data/glove/query.fvecs";
        gtI = "/workspace/data/glove/idx.ivecs";
        gtD = "/workspace/data/glove/dis.fvecs";
    }
    else if(param1 == "text"){
        db = "/workspace/data/text/text10M.fvecs";
        query = "/workspace/data/text/query.fvecs";
        gtI = "/workspace/data/text/idx.ivecs";
        gtD = "/workspace/data/text/dis.fvecs";
    }
    else{
        printf("Your dataset name is illegal\n");
        return 0;
    }

	omp_set_num_threads(16);
    double t0 = elapsed();
    
    // this is typically the fastest one.
    const char* index_key = "IVF1024,Flat";

    faiss::Index* index;

    size_t d;

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float* xt = fvecs_read(db.c_str(), &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0,
               index_key,
               d);
        if(param1 == "sift1M" || param1 == "sift10M" || param1 == "deep10M" || param1 == "gist" || param1 == "spacev")
            index = faiss::index_factory(d, index_key);
        else
            index = faiss::index_factory(d, index_key
            ,faiss::METRIC_INNER_PRODUCT
            );

        // index->set_tune_mode();
        // if(DC(faiss::IndexIVF)){
        //     printf("Output tune type: %d %d\n", index->tune, ix->quantizer->tune);
        // }
        
        printf("Output index type: %d\n", index->type);

        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        // nt = 500000;

        index->set_tune_mode();
        index->train(nt, xt);
        index->set_tune_off();
        delete[] xt;
        std::string filenameIn = "./trained_index/";
        filenameIn += param1;
        filenameIn += "_IVF1024,Flat_trained.index";
        faiss::write_index(index, filenameIn.c_str());
    }

    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        float* xb = fvecs_read(db.c_str(), &d2, &nb);
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
        xq = fvecs_read(query.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k;                // nb of results per query in the GT
    faiss::Index::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read(gtI.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::Index::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    size_t kk;
    float *gt_v;

    {
        printf("[%.3f s] Loading groud truth vector\n", elapsed() - t0);

        size_t nq3;
        gt_v = fvecs_read(gtD.c_str(), &kk, &nq3);
        assert(kk == k || !"gt diatance does not have same dimension as gt IDs");
        assert(nq3 == nq || !"incorrect nb of ground truth entries");
    }

    size_t topk = k;
    size_t max_topk = k;
    // Run error profile system
    {
        printf("[%.3f s] Preparing error profile system criterion 100-recall at 100 "
               "criterion, with k=%ld nq=%ld\n",
               elapsed() - t0,
               k,
               nq);
        faiss::Error_sys err_sys(index, nq , k);

        err_sys.set_gt(gt_v, gt);
        printf("[%.3f s] Start error profile system training\n",
               elapsed() - t0);
        err_sys.sys_train(ts, xq);
        printf("[%.3f s] Finish error profile system training\n",
               elapsed() - t0);

        std::vector<float> D;
        std::vector<int64_t> I;
        std::vector<float> acc;
        size_t demo_size = ses;
        topk = input_k;
        // Set query topk val
        err_sys.set_topk(topk);
        D.resize(demo_size * k);
        I.resize(demo_size * k);
        std::vector<float> accs = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
        // Set required time(ms)
        for(int i = 0; i<demo_size+ts;i++){
            int index = i%accs.size();
            acc.push_back(accs[index]);
        }
        
        err_sys.set_queries(demo_size, xq, acc.data(), ts+ses);
        printf("[%.3f s] Start error profile system search\n",
               elapsed() - t0);
        t0 = elapsed();
        if(DC(faiss::IndexIVF)){
            ix->t->profile = true;
        }
        double tv0 = elapsed();
        std::vector<float> times;
        for(int i = ts; i<ts+ses ;i++){
            t0 = elapsed();
            err_sys.time_search(D.data(), I.data(), i, 1);
            double t1 = elapsed();
            float latency = t1 - t0;
            times.push_back(latency);
        }
        printf("Finish error profile system search: %.3f\n",
               elapsed() - tv0);

        if(DC(faiss::IndexIVF)){
            /// Store the optimal and ELP's nprobe into logs
            std::stringstream ss;
            ss<<"Effective_time_" << param1<<".log";
            std::string filename = ss.str();

            std::ofstream outfile;
            outfile.open(filename);
            for(int i = ts;i < ts+ses; i++){
                outfile << acc[i] << " " << times[i-ts]*1000 <<std::endl;
            }
        }
    }
    delete[] xq;
    delete[] gt;
    delete[] gt_v;
    delete index;
    return 0;
}