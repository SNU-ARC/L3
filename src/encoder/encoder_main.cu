#include "l3.cuh"


string path_in_r = "pix_r.dat";
string path_in_g = "pix_g.dat";
string path_in_b = "pix_b.dat";

string path_out_r = "l3_encoded.l3";


int main() {
    int byte_count = 0;

    // Data input pointers
    char *cpu_in_r, *cpu_in_g, *cpu_in_b;
    void *dev_in_r, *dev_in_g, *dev_in_b;

    // Data output pointers
    unsigned char *cpu_out_r, *cpu_out_g, *cpu_out_b;
    void *dev_out_r, *dev_out_g, *dev_out_b;

    // Patch offset pointers
    unsigned char *cpu_offset_r, *cpu_offset_g, *cpu_offset_b;
    void *dev_offset_r, *dev_offset_g, *dev_offset_b;

    // Uncompressed data size
    size_t size_in = sizeof(char) * wd * ht;

    // Compressed / not packed data size
    size_t size_out = sizeof(char) * (num_wd * (shard + 2)) * ht;

    // Compressed row size
    size_t size_offset = sizeof(char) * num_wd * (num_ht * shard);

    cudaStream_t stream;

    // Memory allocation
    cudaStreamCreate(&stream);

    malloc_gpu(&dev_in_r, &dev_in_g, &dev_in_b, size_in, size_in, size_in);
    malloc_gpu(&dev_out_r, &dev_out_g, &dev_out_b, size_out, size_out, size_out);
    malloc_gpu(&dev_offset_r, &dev_offset_g, &dev_offset_b, size_offset, size_offset, size_offset);

    malloc_cpu((void **)&cpu_in_r, (void **)&cpu_in_g, (void **)&cpu_in_b, size_in, size_in, size_in);
    malloc_cpu((void **)&cpu_offset_r, (void **)&cpu_offset_g, (void **)&cpu_offset_b, size_offset, size_offset, size_offset);

    file_read(path_in_r, path_in_g, path_in_b, &cpu_in_r, &cpu_in_g, &cpu_in_b, size_in, size_in, size_in);

    // Data copy from CPU to GPU
    data_copy(dev_in_r, dev_in_g, dev_in_b,
            (void *)cpu_in_r, (void *)cpu_in_g, (void *)cpu_in_b,
            size_in, size_in, size_in, cudaMemcpyHostToDevice);

    malloc_cpu((void **)&cpu_out_r, (void **)&cpu_out_g, (void **)&cpu_out_b, size_out, size_out, size_out);

    dim3 threadsPerBlock(num_wd, num_ht);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Encoding R, G, and B
    encoder<<<1, threadsPerBlock, 0, stream>>>((unsigned char *)dev_in_r, (unsigned char *)dev_out_r, (unsigned char *)dev_offset_r);
    encoder<<<1, threadsPerBlock, 0, stream>>>((unsigned char *)dev_in_g, (unsigned char *)dev_out_g, (unsigned char *)dev_offset_g);
    encoder<<<1, threadsPerBlock, 0, stream>>>((unsigned char *)dev_in_b, (unsigned char *)dev_out_b, (unsigned char *)dev_offset_b);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timediff_cuda = 0;
    cudaEventElapsedTime(&timediff_cuda, start, stop);

    // Compressed data copy from GPU to CPU
    data_copy((void *)cpu_out_r, (void *)cpu_out_g, (void *)cpu_out_b,
            dev_out_r, dev_out_g, dev_out_b,
            size_out, size_out, size_out, cudaMemcpyDeviceToHost);

    // Compressed row size copy from GPU to CPU
    data_copy((void *)cpu_offset_r, (void *)cpu_offset_g, (void *)cpu_offset_b,
            dev_offset_r, dev_offset_g, dev_offset_b,
            size_offset, size_offset, size_offset, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size_offset; i++) {
        byte_count += cpu_offset_r[i];
        byte_count += cpu_offset_g[i];
        byte_count += cpu_offset_b[i];
    }

    printf("encoded time: %f ms, byte: %d bytes\n", timediff_cuda, byte_count);

    unsigned short part_offset_r[num_wd * num_ht];
    unsigned short part_offset_g[num_wd * num_ht];
    unsigned short part_offset_b[num_wd * num_ht];

    unsigned char compressed_data_r[size_out];
    unsigned char compressed_data_g[size_out];
    unsigned char compressed_data_b[size_out];

    memset(compressed_data_r, 0, size_out);
    memset(compressed_data_g, 0, size_out);
    memset(compressed_data_b, 0, size_out);

    int idx_r = 0;
    int idx_g = 0;
    int idx_b = 0;

    // Calculate patch offset size from compressed row size
    for (int y = 0; y < num_ht; y++) {
        for (int x = 0; x < num_wd; x++) {
            int idx = y * num_wd + x;

            unsigned short arr_sum_r = 0;
            unsigned short arr_sum_g = 0;
            unsigned short arr_sum_b = 0;

            for (int i = 0; i < shard; i++) {
                int idx2 = ((y * shard) + i) * ((shard + 2) * num_wd) + (x * (shard + 2));
                int idx3 = ((y * shard) + i) * num_wd + x;

                memcpy((void *)&compressed_data_r[idx_r], (void *)&cpu_out_r[idx2], cpu_offset_r[idx3]);
                memcpy((void *)&compressed_data_g[idx_g], (void *)&cpu_out_g[idx2], cpu_offset_g[idx3]);
                memcpy((void *)&compressed_data_b[idx_b], (void *)&cpu_out_b[idx2], cpu_offset_b[idx3]);

                idx_r += cpu_offset_r[idx3];
                idx_g += cpu_offset_g[idx3];
                idx_b += cpu_offset_b[idx3];

                arr_sum_r += cpu_offset_r[idx3];
                arr_sum_g += cpu_offset_g[idx3];
                arr_sum_b += cpu_offset_b[idx3];
            }
            part_offset_r[idx] = arr_sum_r;
            part_offset_g[idx] = arr_sum_g;
            part_offset_b[idx] = arr_sum_b;
        }
    }

    // Write L3 format data to storage
    file_write(path_out_r,compressed_data_r, compressed_data_g, compressed_data_b,
               part_offset_r, part_offset_g, part_offset_b,
               idx_r, idx_g, idx_b);

    // Memory free
    free_gpu(dev_in_r, dev_in_g, dev_in_b);
    free_gpu(dev_out_r, dev_out_g, dev_out_b);
    free_gpu(dev_offset_r, dev_offset_g, dev_offset_b);

    free_cpu((void *)cpu_in_r, (void *)cpu_in_g, (void *)cpu_in_b);
    free_cpu((void *)cpu_offset_r, (void *)cpu_offset_g, (void *)cpu_offset_b);

    cudaStreamDestroy(stream);

    free_cpu((void *)cpu_out_r, (void *)cpu_out_g, (void *)cpu_out_b);

    return 0;
}

