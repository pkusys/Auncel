// Copyright (c) Zili Zhang.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <utility>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

using vp = std::pair<int *, float *>;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float* fvecs_read(const char* fname, int* d_out, int* n_out) {
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
    int sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    int n = sz / ((d + 1) * 4);
    int total_size = n * (d + 1);

    *d_out = d;
    *n_out = n;
    printf("Read file from %s, n = %d, d = %d\n", fname, n , d);
    float* x = new float[total_size];
    int nr = fread(x, sizeof(float), total_size, f);
    assert(nr == total_size || !"could not read whole file");

    // shift array to remove row headers
    for (int i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, int* d_out, int* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

void fvecs_write(const char* fname, float * x, int d, int n){
    FILE* f = fopen(fname, "wb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int nw = 0;
    for (int i = 0; i < n; i++){
        nw += fwrite(&d , sizeof(int), 1, f);
        nw += fwrite(x , sizeof(float), d, f);
    }
    printf("Write finished, wite %d\n", nw);
    fclose(f);
}

void ivecs_write(const char* fname, int * x, int d, int n){
    FILE* f = fopen(fname, "wb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int nw = 0;
    for (int i = 0; i < n; i++){
        nw += fwrite(&d , sizeof(int), 1, f);
        nw += fwrite(x , sizeof(int), d, f);
    }
    printf("Write finished, wite %d\n", nw);
    fclose(f);
}

void merge_sort(vp a1, vp a2, vp res){
    int d = 100, n = 10000;
    for (int i = 0; i < n; i ++){
        int ind1 = 0, ind2 = 0, ind = 0;
        while(ind1 <= d && ind2 <= d){
            if (a1.second[ind1 + i*d] < a2.second[ind2 + i*d]){
                res.second[ind + i*d] = a1.second[ind1 + i*d];
                res.first[ind + i*d] = a1.first[ind1 + i*d];
                ind++;
                ind1++;
            }
            else{
                res.second[ind + i*d] = a2.second[ind2 + i*d];
                res.first[ind + i*d] = a2.first[ind2 + i*d];
                ind++;
                ind2++;
            }
            if (d == ind)
                break;
        }
    }
}

vp process_file(std::string name){
    std::string disfile = name + "dis.fvecs";
    std::string idxfile = name + "idx.ivecs";
    int d,n;
    float *dis = fvecs_read(disfile.c_str(), &d, &n);
    int d1, n1;
    int *idx = ivecs_read(idxfile.c_str(), &d1, &n1);
    assert(d==d1 && n==n1);
    vp res;
    res.first = idx;
    res.second = dis;
    return res;
}

void Del(vp v){
    delete[] v.first;
    delete[] v.second;
}


int main(int argc, char **argv){
    double t0 = elapsed();
    if(argc - 1 != 3){
        printf("You at least need three params: input dir , output file and input files's No.\n");
        return 0;
    }
    std::string file_dir = argv[1];
    std::string output_file = argv[2];
    std::string number_str = argv[3];
    int number = std::stoi(number_str);
    assert(number >= 2);
    vp p1 = process_file(file_dir + "0");
    vp p2 = process_file(file_dir + "1");
    int n = 10000, d = 100;
    int *resi = new int[n*d];
    float * resf = new float[n*d];
    vp res;
    res.first = resi;
    res.second = resf;
    merge_sort(p1, p2, res);
    printf("Reduce time: %.3f\n", elapsed() - t0);
    Del(p1);
    Del(p2);
    Del(res);
    return 0;
}