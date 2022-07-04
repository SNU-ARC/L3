#include "l3.cuh"

string path_in_r = "l3_encoded.l3";

int main() {
    // Data input pointers
    unsigned char *cpu_in;
    void *dev_in;

    // Data output pointers
    unsigned char *cpu_out;
    void *dev_out;

    // Patch offset pointers
    unsigned short *offset_cpu_in_r, *offset_cpu_in_g, *offset_cpu_in_b;
    int *offset_cpu_out_r, *offset_cpu_out_g, *offset_cpu_out_b;
    void *offset_dev_out_r, *offset_dev_out_g, *offset_dev_out_b;

    // Decompressed image size
    size_t size_out = sizeof(char) * wd * ht * 3;
    // Total patch offset size of each channel
    size_t size_offset = sizeof(short) * num_wd * num_ht;
    size_t size_offset_int = sizeof(int) * (num_wd * num_ht + 1);

    offset_cpu_in_r = (unsigned short *)malloc(size_offset);
    offset_cpu_in_g = (unsigned short *)malloc(size_offset);
    offset_cpu_in_b = (unsigned short *)malloc(size_offset);

    offset_cpu_out_r = (int *)malloc(size_offset_int);
    offset_cpu_out_g = (int *)malloc(size_offset_int);
    offset_cpu_out_b = (int *)malloc(size_offset_int);

    malloc_gpu(&offset_dev_out_r, &offset_dev_out_g, &offset_dev_out_b, size_offset_int, size_offset_int, size_offset_int);

    cudaStream_t stream;

    unsigned int size_file;
    get_file_size(path_in_r, &size_file);

    cudaMalloc(&dev_in, size_file);

    cudaMalloc(&dev_out, size_out);
    cpu_out = (unsigned char *)malloc(size_out);

    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cpu_in = (unsigned char *)malloc(size_file);

    // Read data from storage to memory (file array)
    file_read_single(path_in_r, (char **)&cpu_in, size_file);

    // Copy patch offset of R, G, and B from file to independent array
    memcpy((void *)offset_cpu_in_r, (void *)&cpu_in[13], size_offset);
    memcpy((void *)offset_cpu_in_g, (void *)&cpu_in[13 + (num_wd * num_ht) * 2], size_offset);
    memcpy((void *)offset_cpu_in_b, (void *)&cpu_in[13 + (num_wd * num_ht) * 4], size_offset);

    // Accumulation sum to restore the patch offset
    offset_cpu_out_r[0] = 0;
    offset_cpu_out_g[0] = 0;
    offset_cpu_out_b[0] = 0;
    for (int i = 0; i < num_wd * num_ht + 1; i++) {
        offset_cpu_out_r[i + 1] = offset_cpu_in_r[i] + offset_cpu_out_r[i];
        offset_cpu_out_g[i + 1] = offset_cpu_in_g[i] + offset_cpu_out_g[i];
        offset_cpu_out_b[i + 1] = offset_cpu_in_b[i] + offset_cpu_out_b[i];
    }

    // Patch offset copy from CPU to GPU
    data_copy(offset_dev_out_r, offset_dev_out_g, offset_dev_out_b,
              (void *)offset_cpu_out_r, (void *)offset_cpu_out_g, (void *)offset_cpu_out_b,
              size_offset_int, size_offset_int, size_offset_int, cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    dim3 blocks(num_wd, num_ht);
    dim3 threads(shard, 1);

    cudaMemcpyAsync(dev_in, cpu_in, size_file, cudaMemcpyHostToDevice, stream);

    // Decoder operation on GPU
    decoder<<<blocks, threads, 0, stream>>>((unsigned char *)dev_in,
                                            (int *)offset_dev_out_r, (int *)offset_dev_out_g, (int *)offset_dev_out_b,
                                            (unsigned char *)dev_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timediff_cuda = 0;
    cudaEventElapsedTime(&timediff_cuda, start, stop);

    // Copy decompressed data from GPU to CPU
    cudaMemcpy((void *)cpu_out, dev_out, size_out, cudaMemcpyDeviceToHost);

    printf("decoded time: %f ms\n", timediff_cuda);

    cudaStreamDestroy(stream);

    cudaFree(dev_in);
    cudaFree(dev_out);
    free_gpu(offset_dev_out_r, offset_dev_out_g, offset_dev_out_b);

    free(cpu_in);
    free((void *)cpu_out);
    free_cpu((void *)offset_cpu_in_r, (void *)offset_cpu_in_g, (void *)offset_cpu_in_b);

    free(offset_cpu_out_r);
    free(offset_cpu_out_g);
    free(offset_cpu_out_b);
}

