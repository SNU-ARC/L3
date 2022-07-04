#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>

#include <sys/time.h>
#include <unistd.h>

using namespace std;

#define wd      2048
#define ht      1080

#define shard   64
#define num_wd  (int)(wd / shard)
#define num_ht  (int)(ht / shard)
#define pad     (int)((shard + 2 + 4) / 4)
#define multi   1


inline void file_read(string path_r, string path_g, string path_b, 
        char **r, char **g, char **b,
        size_t size_r, size_t size_g, size_t size_b) {

    ifstream input_r(path_r, ifstream::binary);
    ifstream input_g(path_g, ifstream::binary);
    ifstream input_b(path_b, ifstream::binary);

    input_r.read(*r, size_r);
    input_g.read(*g, size_g);
    input_b.read(*b, size_b);

    input_r.close();
    input_g.close();
    input_b.close();
}


inline void file_read_single(string path_r, char **r, size_t size_r) {

    ifstream input_r(path_r, ifstream::binary);
    input_r.read(*r, size_r);
    input_r.close();
}


inline void get_file_size(string path_r, unsigned int *size_r) {

    ifstream input_r(path_r, ifstream::binary);
    input_r.seekg(0, ios::end);
    *size_r = input_r.tellg();
    input_r.close();
}


inline void file_write(string out_path_r,
        unsigned char *r, unsigned char *g, unsigned char *b,
        unsigned short *h_r, unsigned short *h_g, unsigned short *h_b,
        int size_r, int size_g, int size_b) {

    char signal[4] = {76, 76, 76, 46};
    int wd_ht[2] = {wd, ht};
    char patch_size = shard;

    ofstream output_r(out_path_r, ios::out | ios::binary);

    output_r.write(signal, 4 * sizeof(char));
    output_r.write((char *)wd_ht, 2 * sizeof(int));
    output_r.write(&patch_size, sizeof(char));

    int size_h = sizeof(unsigned short) * num_ht * num_wd;

    output_r.write((char *)h_r, size_h);
    output_r.write((char *)h_g, size_h);
    output_r.write((char *)h_b, size_h);

    output_r.write((char *)r, size_r);
    output_r.write((char *)g, size_g);
    output_r.write((char *)b, size_b);

    output_r.close();
}


inline void malloc_gpu(void **r, void **g, void **b,
        size_t size_r, size_t size_g, size_t size_b) {

    cudaMalloc(r, size_r);
    cudaMalloc(g, size_g);
    cudaMalloc(b, size_b);
}


inline void malloc_cpu(void **r, void **g, void **b,
        size_t size_r, size_t size_g, size_t size_b) {

    *r = (void *)malloc(size_r);
    *g = (void *)malloc(size_g);
    *b = (void *)malloc(size_b);
}


inline void free_gpu(void *r, void *g, void *b) {

    cudaFree(r);
    cudaFree(g);
    cudaFree(b);
}


inline void free_cpu(void *r, void *g, void *b) {

    free(r);
    free(g);
    free(b);
}


inline void data_copy(void *dst_r, void* dst_g, void* dst_b,
        void *src_r, void* src_g, void* src_b,
        size_t size_r, size_t size_g, size_t size_b, cudaMemcpyKind direct) {

    cudaMemcpy(dst_r, src_r, size_r, direct);
    cudaMemcpy(dst_g, src_g, size_g, direct);
    cudaMemcpy(dst_b, src_b, size_b, direct);
}


__device__ inline char paeth_filter(unsigned char a, unsigned char b, unsigned char c) {
    // Custom Paeth filter
    // a: Top-left pixel
    // b: Top pixel
    // c: Top-right pixel

    // Step 1: Calculate ref. value
    int p = (int)(a + c - b);

    // Step 2: Return nearest pixel from ref. value
    if (a == 0) {
        int pb = abs(p - b);
        int pc = abs(p - c);

        if (pb <= pc) return b;
        else          return c;
    } else if (c == 0) {
        int pa = abs(p - a);
        int pb = abs(p - b);

        if (pa <= pb) return a;
        else          return b;
   
    } else {
        int pa = abs(p - a);
        int pb = abs(p - b);
        int pc = abs(p - c);

        if (pa <= pb && pa <= pc) return a;
        else if (pa <= pc)        return b;
        else                      return c;
    }

    return -1;
}


// Encoder
__device__ void paeth_loop_enc(unsigned char* unpack, int pos, int idx, char *paeth);
__device__ void bd_loop_enc(char *paeth, char* bd);
__device__ void bitpack_loop_enc(char* bd, unsigned char* pack, unsigned char* off);
__global__ void encoder(unsigned char* origin, unsigned char* packed, unsigned char* offset);


// Decoder
__device__ int bd_loop_dec(unsigned char* bd, char* paeth);
__device__ void paeth_loop_dec(char* paeth, int i, int idx, unsigned char* unpack);
__global__ void decoder(unsigned char* input, int* offset_r, int* offset_g, int* offset_b,  unsigned char* output);

