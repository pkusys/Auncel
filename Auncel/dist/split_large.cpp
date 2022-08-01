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

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

using vf = std::vector<float*>;

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float* fvecs_read(const char* fname, int64_t* d_out, int64_t* n_out) {
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
    int64_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    int64_t n = sz / ((d + 1) * 4);
    int64_t total_size = n * (d + 1);

    *d_out = d;
    *n_out = n;
    printf("Read file from %s, n = %d, d = %d\n", fname, n , d);
    float* x = new float[total_size];
    int64_t nr = fread(x, sizeof(float), total_size, f);
    assert(nr == total_size || !"could not read whole file");

    // shift array to remove row headers
    for (int64_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

float* fbin_read(const char* fname, int64_t* d_out, int64_t* n_out) {
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
    int64_t total_size = int64_t(d) * n;
    size_t slice_size = total_size/100;
    float* x = new float[total_size/2];
    int64_t nr = 0;
    // fseek(f, 192000000008, SEEK_SET);
    for (int i = 0; i<100/2; i++){
        nr += fread(x, sizeof(float), slice_size, f);
        x+=slice_size;
        printf("Read %d/100 slice done...\n", i);
    }
    // int64_t nr = fread(x, sizeof(float), total_size, f);
    printf("Read finished, read %lld\n", nr);
    assert(nr == total_size/2 || !"could not read whole file");
    fclose(f);
    return x - (total_size)/2;
}

void fbin_write(const char* fname, float * x, int d, int n) {
    FILE* f = fopen(fname, "wb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int64_t total_size = int64_t(d) * n;
    size_t slice_size = total_size/100;
    printf("Write file ...\n");
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&d, sizeof(int), 1, f);
    int64_t nr = 0;
    for (int i = 0; i < 100 ;i++){
        nr += fwrite(x , sizeof(float), slice_size, f);
        x += slice_size;
        printf("Write %d/100 slice done...\n", i);
    }
    // nr += fwrite(x , sizeof(float), total_size, f);
    printf("Write finished, wite %lld\n", nr);
    fclose(f);
    // printf("Write finished, wite %lld\n", nr);
    return;
}



// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, int64_t* d_out, int64_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}


int main(int argc,char **argv) {
    if(argc - 1 != 2){
        printf("You at least need tow params: input file and output dir\n");
        return 0;
    }
    std::string file_name = argv[1];
    std::string file_dir = argv[2];
    std::string file1 = file_dir + "0db.fbin";
    std::string file2 = file_dir + "1db.fbin";
    int64_t n,d;
    double t0 = elapsed();
    float *x = fbin_read(file_name.c_str(), &d, &n);
    printf("Write time: %.3f", elapsed() - t0);
    t0 = elapsed();
    fbin_write(file1.c_str(), x, d, n/2);
    delete[] x;
    return 0;
}