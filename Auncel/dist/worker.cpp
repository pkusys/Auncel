// Copyright (c) Zili Zhang.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<errno.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<unistd.h>

/// CPP part
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>

#include <sys/time.h>

#include "../IndexIVF.h"
#include "../profile.h"
#include "../AutoTune.h"

#include <omp.h>

#include<iostream>
#include<fstream>

#define MAXLINE 4096
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

int main(int argc, char** argv){
    omp_set_num_threads(16);
    double t0 = elapsed();
    
    const char* index_key = "IVF1024,Flat";
    faiss::Index* index;
    size_t d;

    std::string db, query, gtI, gtD;
    db = "/workspace/data/dis/deep10M/db0.fvecs";
    query = "/workspace/data/dis/deep10M/query.fvecs";
    gtI = "/workspace/data/dis/deep10M/idx.ivecs";
    gtD = "/workspace/data/dis/deep10M/dis.fvecs";

    printf("[%.3f s] Loading train set\n", elapsed() - t0);

    size_t nt;
    float* xt = fvecs_read(db.c_str(), &d, &nt);

    printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
            elapsed() - t0,
            index_key,
            d);
    index = faiss::index_factory(d, index_key);
    DC(faiss::IndexIVF);

    size_t k;                // nb of results per query in the GT
    faiss::Index::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors
    size_t nq;
    float* xq;
    float *gt_v;
    size_t topk;
    size_t max_topk;
    faiss::Error_sys err_sys;

    int  listenfd, connfd;
    struct sockaddr_in  servaddr;
    char  buff[4096];
    int  n;
    bool is_trained = false;

    if( (listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1 ){
        printf("create socket error: %s(errno: %d)\n",strerror(errno),errno);
        return 0;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(3456);

    if( bind(listenfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1){
        printf("bind socket error: %s(errno: %d)\n",strerror(errno),errno);
        return 0;
    }

    if( listen(listenfd, 10) == -1){
        printf("listen socket error: %s(errno: %d)\n",strerror(errno),errno);
        return 0;
    }

    printf("======waiting for master's request======\n");
    while(1){
        if( (connfd = accept(listenfd, (struct sockaddr*)NULL, NULL)) == -1){
            printf("accept socket error: %s(errno: %d)\n",strerror(errno),errno);
            continue;
        }
        n = recv(connfd, buff, MAXLINE, 0);
        printf("recv %d chars\n", n);
        buff[n] = '\0';
        std::string s = buff;
        printf("recv msg from master: %s\n", buff);
        std::cout << n << " " << s << std::endl;

        if(s == "train" && !is_trained){
            printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
            index->set_tune_mode();
            index->train(nt, xt);
            index->set_tune_off();
            delete[] xt;

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
            

            {
                printf("[%.3f s] Loading queries\n", elapsed() - t0);

                size_t d2;
                xq = fvecs_read(query.c_str(), &d2, &nq);
                assert(d == d2 || !"query does not have same dimension as train set");
            }

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

            {
                printf("[%.3f s] Loading groud truth vector\n", elapsed() - t0);

                size_t nq3;
                gt_v = fvecs_read(gtD.c_str(), &kk, &nq3);
                assert(kk == k || !"gt diatance does not have same dimension as gt IDs");
                assert(nq3 == nq || !"incorrect nb of ground truth entries");
            }

            topk = k;
            max_topk = k;
            printf("[%.3f s] Preparing error profile system criterion 100-recall at 100 "
               "criterion, with k=%ld nq=%ld\n",
               elapsed() - t0,
               k,
               nq);
            faiss::Error_sys tmp_err_sys(index, nq , k);
            err_sys = tmp_err_sys;

            err_sys.set_gt(gt_v, gt);
            printf("[%.3f s] Start error profile system training\n",
                elapsed() - t0);
            err_sys.sys_train(2000, xq);
            printf("[%.3f s] Finish error profile system training\n",
                elapsed() - t0);

            
            
            is_trained = true;
            std::string su = "success";
            send(connfd, su.c_str(), strlen(su.c_str()), 0);
            close(connfd);
        }
        else if(s == "train"){

            printf("Index has already trained\n");
            std::string su = "success";
            send(connfd, su.c_str(), strlen(su.c_str()), 0);
            close(connfd);
        }
        else if(s == "search"){
            if(is_trained){
                std::vector<float> D;
                std::vector<int64_t> I;
                std::vector<float> acc;
                size_t demo_size = 8000;
                size_t ts = 2000;
                // Set query topk val
                err_sys.set_topk(topk);
                D.resize(demo_size * k);
                I.resize(demo_size * k);
                // Set required recalls
                std::vector<float> accs = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6 , 0.7, 0.8, 0.9};
                for(int i = 0; i<demo_size+ts;i++){
                    int index = i%accs.size();
                    acc.push_back(accs[index]);
                }
                
                err_sys.set_queries(demo_size, xq, acc.data(), ts + demo_size);
                printf("[%.3f s] Start error profile system search\n",
                    elapsed() - t0);
                t0 = elapsed();
                err_sys.search(D.data(), I.data(), ts);
                printf("Finish error profile system search: %.3f\n",
                    elapsed() - t0);
                
                // save result into files
                std::vector<int> int_I(demo_size * topk);
                for(int j = 0;j<demo_size*topk;j++)
                    int_I[j] = I[j];
                FILE* f = fopen("./idx.ivecs", "wb");
                for(size_t i = 0; i < demo_size; i++){
                    fwrite(&topk, sizeof(int), 1, f);
                    fwrite(int_I.data() + (topk * i), sizeof(int), topk, f);
                }
                fclose(f);

                FILE* tf = fopen("./dis.fvecs", "wb");
                for(size_t i = 0; i < demo_size; i++){
                    fwrite(&topk, sizeof(int), 1, tf);
                    fwrite(D.data() + (topk * i), sizeof(float), topk, tf);
                }
                fclose(tf);
                
                char  buffer[MAXLINE];
                FILE *fp, *fp2;
                int len;
                fp = fopen("./idx.ivecs", "rb");
                while(!feof(fp)){
                    len = fread(buffer, 1, sizeof(buffer), fp);
                    if(len != send(connfd, buffer, len, 0)){
                        printf("slave send file failed\n");
                        break;
                    }
                }
                usleep(100000);
                std::string file_end = "file done";
                send(connfd, file_end.c_str(), strlen(file_end.c_str()), 0);
                usleep(100000);
                fp2 = fopen("./dis.fvecs", "rb");
                while(!feof(fp2)){
                    len = fread(buffer, 1, sizeof(buffer), fp2);
                    if(len != send(connfd, buffer, len, 0)){
                        printf("slave send file failed\n");
                        break;
                    }
                }
                usleep(100000);
                send(connfd, file_end.c_str(), strlen(file_end.c_str()), 0);
                usleep(100000);
                std::string su = "success";
                send(connfd, su.c_str(), strlen(su.c_str()), 0);
                // usleep(100000);
                fclose(fp);
                fclose(fp2);
                close(connfd);
            }
            else{
                std::string su = "index has not trained";
                send(connfd, su.c_str(), strlen(su.c_str()), 0);
                close(connfd);
            }
        }
    } 
    close(listenfd);  
    delete[] xq;
    delete[] gt;
    delete[] gt_v;
    return 0;
}