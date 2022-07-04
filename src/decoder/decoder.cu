#include "l3.cuh"

__device__ int bd_loop_dec(unsigned char* bd, char* paeth) {
    // Step 1: Restore deltas from bits
    int nbit_entry = (bd[0] & 0x0F);
    char base = ((bd[0] & 0xF0) >> 4) + ((bd[1] & 0x0F) << 4);

    int sbit = 12 + threadIdx.x * nbit_entry;
    int ebit = 12 + (threadIdx.x + 1) * nbit_entry - 1;

    int sbound = floorf(sbit / 8);
    int ebound = floorf(ebit / 8);

    int sstart = sbit - sbound * 8;
    int estart = ebit - ebound * 8;

    if (sbound == ebound) {
        unsigned char flag = 0;
        for (int j = 0; j < nbit_entry; j++) {
            flag += (1 << j);
        }
        flag = flag << sstart;

        // Step 2: Calculate deltas + base
        paeth[threadIdx.x] = ((bd[sbound] & flag) >> sstart) + base;
    } else {
        int nbit_entry_a = 8 - sstart;
        int nbit_entry_b = estart + 1;

        unsigned char flag_a = 0;
        for (int j = 0; j < nbit_entry_a; j++) {
            flag_a += ((unsigned char)128 >> j);
        }

        unsigned char flag_b = 0;
        for (int j = 0; j < nbit_entry_b; j++) {
            flag_b += (1 << j);
        }

        // Step 2: Calculate deltas + base
        paeth[threadIdx.x] = ((bd[sbound] & flag_a) >> sstart) + ((bd[ebound] & flag_b) << nbit_entry_a) + base;
    }

    return (int)roundf((12 + nbit_entry * 64) / 8 + 0.5);
}


__device__ void paeth_loop_dec(char* paeth, int i, int idx, unsigned char* unpack) {
    if (i == 0) {
        // Skip first row
        unpack[idx + 3 * threadIdx.x] = paeth[threadIdx.x];
    } else {
        if (threadIdx.x == 0) {
            unpack[idx + 3 * threadIdx.x] = paeth[threadIdx.x] + paeth_filter(0,
                                                                        unpack[idx + 3 * (threadIdx.x - wd)],
                                                                        unpack[idx + 3 * (threadIdx.x - wd + 1)]);
        } else if (threadIdx.x == (shard - 1)) {
            unpack[idx + 3 * threadIdx.x] = paeth[threadIdx.x] + paeth_filter(unpack[idx + 3 * (threadIdx.x - wd - 1)],
                                                                        unpack[idx + 3 * (threadIdx.x - wd)],
                                                                        0);
        } else {
            unpack[idx + 3 * threadIdx.x] = paeth[threadIdx.x] + paeth_filter(unpack[idx + 3 * (threadIdx.x - wd - 1)],
                                                                        unpack[idx + 3 * (threadIdx.x - wd)],
                                                                        unpack[idx + 3 * (threadIdx.x - wd + 1)]);
        }
    }
}


__global__ void decoder(unsigned char* input, int* offset_r, int* offset_g, int* offset_b,  unsigned char* output) {
    int offset_idx = blockIdx.y * num_wd + blockIdx.x;

    int header_size = sizeof(short) * (num_wd * num_ht * 3) + 13;

    // Start offset of compressed R data
    int pt_offset_r = offset_r[offset_idx] + header_size;
    // Start offset of compressed G data
    int pt_offset_g = offset_g[offset_idx] + header_size + offset_r[num_wd * num_ht];
    // Start offset of compressed B data
    int pt_offset_b = offset_b[offset_idx] + header_size + offset_r[num_wd * num_ht] + offset_g[num_wd * num_ht];

    char paeth_r[shard], paeth_g[shard], paeth_b[shard];

    for (int i = 0; i < shard; i++) {
        memset(paeth_r, 0, sizeof(char) * shard);
        memset(paeth_g, 0, sizeof(char) * shard);
        memset(paeth_b, 0, sizeof(char) * shard);

        int row_off_r = bd_loop_dec(&input[pt_offset_r], paeth_r);
        int row_off_g = bd_loop_dec(&input[pt_offset_g], paeth_g);
        int row_off_b = bd_loop_dec(&input[pt_offset_b], paeth_b);

        pt_offset_r += row_off_r;
        pt_offset_g += row_off_g;
        pt_offset_b += row_off_b;

        int idx = (blockIdx.y * shard + i) * (wd * 3) + (blockIdx.x * (shard * 3));

        paeth_loop_dec(paeth_r, i, idx, output);
        paeth_loop_dec(paeth_g, i, idx + 1, output);
        paeth_loop_dec(paeth_b, i, idx + 2, output);

        __syncthreads();
    }
}

