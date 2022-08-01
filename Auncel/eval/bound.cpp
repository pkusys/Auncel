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

/* type = 0 : L2, 1 : IP*/
size_t inter_sec(size_t max_topk, const float *gt, size_t topk, const float *I, int type = 0){
    size_t res = 0;
    float t_val = gt[topk-1];
    for(int i = 0; i < topk;i++){
        float c_val = I[i];
        if (c_val <= t_val + 1e-6 && type == 0)
            res++;
        if (c_val >= t_val - 1e-6 && type == 1)
            res++;
    }
    return res;
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/// Command like this: ./knn_script sift1M 100 2000 8000
int main(int argc,char **argv) {
    std::cout << argc << " arguments" <<std::endl;
    if(argc - 1 != 6){
        printf("You should at least input 6 params: the dataset name, train size, query size, topk , error bound and figure id\n");
        return 0;
    }
    std::string p1 = argv[1];
    std::string p2 = argv[2];
    std::string p3 = argv[3];
    std::string p4 = argv[4];
    std::string p5 = argv[5];
    std::string p6 = argv[6];

    int input_k = std::stoi(p4);
    int ts = std::stoi(p2);
    int ses = std::stoi(p3);
    float error_bound = std::stof(p5);
    int figureid = std::stoi(p6);

    if(input_k>100 || input_k <0){
        printf("Input topk must be lower than or equal to 100 and greater than 0\n");
        return 0;
    }
    std::string db, query, gtI, gtD;
    if(p1 == "sift1M"){
        db = "/workspace/data/sift/sift1M.fvecs";
        query = "/workspace/data/sift/1M_query.fvecs";
        gtI = "/workspace/data/sift/idx_1M.ivecs";
        gtD = "/workspace/data/sift/dis_1M.fvecs";
    }
    else if(p1 == "sift10M"){
        db = "/workspace/data/sift/sift10M/sift10M.fvecs";
        query = "/workspace/data/sift/sift10M/query.fvecs";
        gtI = "/workspace/data/sift/sift10M/idx.ivecs";
        gtD = "/workspace/data/sift/sift10M/dis.fvecs";
    }
    else if(p1 == "deep10M"){
        db = "/workspace/data/deep/deep10M.fvecs";
        query = "/workspace/data/deep/query.fvecs";
        gtI = "/workspace/data/deep/idx.ivecs";
        gtD = "/workspace/data/deep/dis.fvecs";
    }
    else if(p1 == "gist"){
        db = "/workspace/data/gist/gist1M.fvecs";
        query = "/workspace/data/gist/query.fvecs";
        gtI = "/workspace/data/gist/idx.ivecs";
        gtD = "/workspace/data/gist/dis.fvecs";
    }
    else if(p1 == "spacev"){
        db = "/workspace/data/spacev/spacev10M.fvecs";
        query = "/workspace/data/spacev/query.fvecs";
        gtI = "/workspace/data/spacev/idx.ivecs";
        gtD = "/workspace/data/spacev/dis.fvecs";
    }
    else if(p1 == "glove"){
        db = "/workspace/data/glove/glove.fvecs";
        query = "/workspace/data/glove/query.fvecs";
        gtI = "/workspace/data/glove/idx.ivecs";
        gtD = "/workspace/data/glove/dis.fvecs";
    }
    else if(p1 == "text"){
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
        if(p1 == "sift1M" || p1 == "sift10M" || p1 == "deep10M" || p1 == "gist" || p1 == "spacev")
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
        filenameIn += p1;
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
        // Set required recalls
        std::vector<float> accs;
        accs.push_back(1. - error_bound);
        for(int i = 0; i<demo_size+ts;i++){
            int index = i%accs.size();
            acc.push_back(accs[index]);
        }
        
        err_sys.set_queries(demo_size, xq, acc.data(), ts+ses);
        printf("[%.3f s] Start error profile system search\n",
               elapsed() - t0);
        t0 = elapsed();
        if(DC(faiss::IndexIVF)){
            ix->t->setparam(figureid);
        }

        std::vector<double> latency;
        for(int i = ts; i<ts+ses ;i++){
            auto tt0 = elapsed();
            err_sys.search(D.data() + k * (i - ts), I.data() + k * (i - ts), i, 1);
            auto tt1 = elapsed();
            latency.push_back(tt1 - tt0);
        }
        printf("Finish error profile system search: %.3f\n",
               elapsed() - t0);

        int type = 0;
        if (p1 == "text")
            type = 1;

        float minf = 1.;
        for (int i = ts; i < ses + ts; i++) {
            minf = std::min(minf, inter_sec(k, &gt_v[i * 100], 
                input_k, D.data() + (i - ts) * k , type)/float(input_k));
        }
        if (minf >= (1 - error_bound))
            printf("Error bound is guaranteed\n\n\n");
        else
            printf("NO NO NO !!! Error bound is not guaranteed, please enlarge top-n \n");

        printf("Error Bound : %f\n", minf);

        // Output the latency to file
        std::stringstream fn;
        fn<<"Auncel_Latency" << "_" << p1 << "_" << input_k << "_" << int(error_bound*100) <<".log";
        std::string filename = fn.str();

        std::ofstream outfile;
        outfile.open(filename);
        for(int i = 0;i < ses; i++){
            outfile << latency[i] << std::endl;
        }
        outfile.close();

    }
    delete[] xq;
    delete[] gt;
    delete[] gt_v;
    delete index;
    return 0;
}