#include "l3.cuh"


__device__ void paeth_loop_enc(unsigned char* unpack, int pos, int idx, char *paeth) {
    if (pos == 0) {
        // Skip first row
        for (int i = 0; i < shard; i++) {
            paeth[i] = unpack[idx + i];
        }
    } else {
        paeth[0] = unpack[idx] - paeth_filter(0, unpack[idx - wd], unpack[idx - wd + 1]);
        for (int i = 1; i < shard-1; i++) {
            paeth[i] = unpack[idx + i] - paeth_filter(unpack[idx + i - wd - 1], unpack[idx + i - wd], unpack[idx + i - wd + 1]);
        }
        paeth[shard-1] = unpack[idx + shard - 1] - paeth_filter(unpack[idx + (shard - 1) - wd - 1], unpack[idx + (shard - 1) - wd], 0);
    }
}


__device__ void bd_loop_enc(char *paeth, char* bd) {

    // Step 1: Find min, max
    int min_arr = paeth[0], max_arr = paeth[0];

    for (int i = 1; i < shard; i++) {
        min_arr = min(paeth[i], min_arr);
        max_arr = max(paeth[i], max_arr);
    }

    // Min value to base
    char base = min_arr;
    float diff = max_arr - base;

    // Step 2: Calculate # of bits per entry
    int nbit = (int)roundf(log2f(diff) + 0.5);

    bd[0] = nbit, bd[1] = base;

    // Step 3: Calculate deltas
    for (int i = 0; i < shard; i++) {
        bd[i + 2] = paeth[i] - base;
    }
}


__device__ void bitpack_loop_enc(char* bd, unsigned char* pack, unsigned char* off) {
    int idx = 0;
    int i = 2;

    // Step 4: Bit-packing
    pack[idx++] = (bd[0] & 0x0F) + ((bd[1] & 0x0F) << 4);
    pack[idx] = ((bd[1] & 0xF0) >> 4);

    switch((int)bd[0]) {
        case 1:
        {
            pack[idx++] += ((bd[2] & 0x01) << 4) + ((bd[3] & 0x01) << 5) + ((bd[4] & 0x01) << 6) + ((bd[5] & 0x01) << 7);

            for (i = 6; i < shard - 2; i += 8) {
                pack[idx++] = (bd[i + 0] & 0x01) + ((bd[i + 1] & 0x01) << 1) + ((bd[i + 2] & 0x01) << 2) +
                              ((bd[i + 3] & 0x01) << 3) + ((bd[i + 4] & 0x01) << 4) + ((bd[i + 5] & 0x01) << 5) +
                              ((bd[i + 6] & 0x01) << 6) + ((bd[i + 7] & 0x01) << 7);
            }

            pack[idx++] = (bd[shard - 2] & 0x01) + ((bd[shard - 1] & 0x01) << 1) + ((bd[shard] & 0x01) << 2) + ((bd[shard + 1] & 0x01) << 3);

            break;
        }
        case 2:
        {
            pack[idx++] += ((bd[2] & 0x03) << 4) + ((bd[3] & 0x03) << 6);

            for (i = 4; i < shard; i+= 4) {
                pack[idx++] = (bd[i] & 0x03) + ((bd[i + 1] & 0x03) << 2) + ((bd[i + 2] & 0x03) << 4) + ((bd[i + 3] & 0x03) << 6);
            }

            pack[idx++] = (bd[shard] & 0x03) + ((bd[shard + 1] & 0x03) << 2);

            break;
        }
        case 3:
        {
            pack[idx++] += ((bd[2] & 0x07) << 4) + ((bd[3] & 0x01) << 7);

            for (i = 3; i < shard - 5; i += 8) {
                pack[idx++] = ((bd[i] & 0x06) >> 1) + ((bd[i + 1] & 0x07) << 2) + ((bd[i + 2] & 0x07) << 5);
                pack[idx++] = (bd[i + 3] & 0x07)  + ((bd[i + 4] & 0x07) << 3) + ((bd[i + 5] & 0x03) << 6);
                pack[idx++] = ((bd[i + 5] & 0x04) >> 2) + ((bd[i + 6] & 0x07) << 1) + ((bd[i + 7] & 0x07) << 4) + ((bd[i + 8] & 0x01) << 7);
            }

            pack[idx++] = ((bd[shard - 5] & 0x06) >> 1) + ((bd[shard - 4] & 0x07) << 2) + ((bd[shard - 3] & 0x07) << 5);
            pack[idx++] = (bd[shard - 2] & 0x07) + ((bd[shard - 1] & 0x07) << 3) + ((bd[shard] & 0x03) << 6);
            pack[idx++] = ((bd[shard] & 0x04) >> 2) + ((bd[shard + 1] & 0x07) << 1);

            break;
        }
        case 4:
        {
            pack[idx++] += ((bd[2] & 0x0F) << 4);

            for (i = 3; i < shard + 1; i += 2) {
                pack[idx++] = (bd[i] & 0x0F) + ((bd[i + 1] & 0x0F) << 4);
            }

            pack[idx++] = (bd[shard + 1] & 0x0F);

            break;
        }
        case 5:
        {
            pack[idx++] += ((bd[2] & 0x0F) << 4);

            for (i = 2; i < shard - 6; i += 8) {
                pack[idx++] = ((bd[i] & 0x10) >> 4) + ((bd[i + 1] & 0x1F) << 1) + ((bd[i + 2] & 0x03) << 6);
                pack[idx++] = ((bd[i + 2] & 0x1C) >> 2) + ((bd[i + 3] & 0x1F) << 3);
                pack[idx++] = (bd[i + 4] & 0x1F) + ((bd[i + 5] & 0x07) << 5);
                pack[idx++] = ((bd[i + 5] & 0x18) >> 3) + ((bd[i + 6] & 0x1F) << 2) + ((bd[i + 7] & 0x01) << 7);
                pack[idx++] = ((bd[i + 7] & 0x1E) >> 1) + ((bd[i + 8] & 0x0F) << 4);
            }
    
            pack[idx++] = ((bd[shard - 6] & 0x10) >> 4) + ((bd[shard - 5] & 0x1F) << 1) + ((bd[shard - 4] & 0x03) << 6);
            pack[idx++] = ((bd[shard - 4] & 0x1C) >> 2) + ((bd[shard - 3] & 0x1F) << 3);
            pack[idx++] = (bd[shard - 2] & 0x1F) + ((bd[shard - 1] & 0x07) << 5);
            pack[idx++] = ((bd[shard - 1] & 0x18) >> 3) + ((bd[shard] & 0x1F) << 2) + ((bd[shard + 1] & 0x01) << 7);
            pack[idx++] = ((bd[shard + 1] & 0x1E) >> 1);

            break;
        }
        case 6:
        {
            pack[idx++] += ((bd[2] & 0x0F) << 4);

            for (i = 2; i < shard - 2; i += 4) {
                pack[idx++] = ((bd[i] & 0x30) >> 4) + ((bd[i + 1] & 0x3F) << 2);
                pack[idx++] = (bd[i + 2] & 0x3F) + ((bd[i + 3] & 0x03) << 6);
                pack[idx++] = ((bd[i + 3] & 0x1C) >> 2) + ((bd[i + 4] & 0x0F) << 4);
            }

            pack[idx++] = ((bd[shard - 2] & 0x30) >> 4) + (bd[shard - 1] & 0x3F << 2);
            pack[idx++] = (bd[shard] & 0x3F) + ((bd[shard + 1] & 0x03) << 6);
            pack[idx++] = ((bd[shard + 1] & 0x1C) >> 2);

            break;
        }
        case 7:
        {
            pack[idx++] += ((bd[2] & 0x0F) << 4);

            for (int i = 2; i < shard - 6; i += 8) {
                pack[idx++] = ((bd[i] & 0x70) >> 4) + ((bd[i + 1] & 0x1F) << 3);
                pack[idx++] = ((bd[i + 1] & 0x60) >> 5) + ((bd[i + 2] & 0x3F) << 2);
                pack[idx++] = ((bd[i + 2] & 0x40) >> 6) + ((bd[i + 3] & 0x7F) << 1);
                pack[idx++] = (bd[i + 4] & 0x7F) + ((bd[i + 5] & 0x01) << 7);
                pack[idx++] = ((bd[i + 5] & 0x7E) >> 1) + ((bd[i + 6] & 0x03) << 6);
                pack[idx++] = ((bd[i + 6] & 0x7C) >> 2) + ((bd[i + 7] & 0x07) << 5);
                pack[idx++] = ((bd[i + 7] & 0x78) >> 3) + ((bd[i + 8] & 0x0F) << 4);
            }        

            pack[idx++] = ((bd[shard - 6] & 0x70) >> 4) + ((bd[shard - 5] & 0x1F) << 3);
            pack[idx++] = ((bd[shard - 5] & 0x60) >> 5) + ((bd[shard - 4] & 0x3F) << 2);
            pack[idx++] = ((bd[shard - 4] & 0x40) >> 6) + ((bd[shard - 3] & 0x7F) << 1);
            pack[idx++] = (bd[shard - 2] & 0x7F) + ((bd[shard - 1] & 0x01) << 7);
            pack[idx++] = ((bd[shard - 1] & 0x7E) >> 1) + ((bd[shard] & 0x03) << 6);
            pack[idx++] = ((bd[shard] & 0x7C) >> 2) + ((bd[shard + 1] & 0x07) << 5);
            pack[idx++] = ((bd[shard + 1] & 0x78) >> 3);

            break;
        }
        default:
            break;
    }

    *off = (unsigned char)(idx);
}

__global__ void encoder(unsigned char* origin, unsigned char* packed, unsigned char* offset) {

    char paeth[shard];
    char bd[shard + 3];

    for (int i = 0; i < shard; i++) {
        memset(paeth, 0, sizeof(char) * shard);
        memset(bd, 0, sizeof(char) * (shard + 2));

        int idx = (threadIdx.y * shard + i) * wd + threadIdx.x * shard;
        int idx2 = (threadIdx.y * shard + i) * ((shard + 2) * num_wd) + threadIdx.x * (shard + 2);
        int idx3 = (threadIdx.y * shard + i) * num_wd + threadIdx.x;

        // Paeth filter
        paeth_loop_enc(origin, i, idx, paeth);

        // BDI
        bd_loop_enc(paeth, bd);

        // Bit packing
        bitpack_loop_enc(bd, &packed[idx2], &offset[idx3]);
    }
}

