/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "pre_sync_cuda.cuh"

// The block cuda file is just a wrapper for the kernels that will be launched in the work
// function
namespace gr {
namespace wifi {
namespace pre_sync_cu {

__global__ void
corr_abs_kernel(cuFloatComplex* in, cuFloatComplex* out, float* mag, int n)
{
    int d = 16;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuFloatComplex m = cuCmulf((in[i + d]), cuConjf(in[i]));

        float cplx_mag = in[i].x * in[i].x + in[i].y * in[i].y;
        mag[i] = cplx_mag;
        out[i] = m;
    }
}


__global__ void mov_avg_cplx_kernel(
    cuFloatComplex* in, float* mag, cuFloatComplex* out, float* cor, int n)
{
    // int d = 16;
    int w = 48;
    int w2 = 64;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {

        cuFloatComplex sum = make_cuFloatComplex(0, 0);
        for (int j = 0; j < w; j++) {
            sum.x += in[i + j].x;
            sum.y += in[i + j].y;
        }

        float fsum = 0;
        for (int j = 0; j < w2; j++) {
            fsum += mag[i + j];
        }

        // __syncthreads();
        // if (i < n-63-16) {
        out[i] = sum;
        // }
        if (fsum == 0)
            cor[i] = 0;
        else
            cor[i] = sqrt(sum.x * sum.x + sum.y * sum.y) / fsum;
    }
}


void exec_corr_abs(cuFloatComplex* in,
                   cuFloatComplex* out,
                   float* mag,
                   int n,
                   int grid_size,
                   int block_size,
                   cudaStream_t stream)
{
    corr_abs_kernel<<<grid_size, block_size, 0, stream>>>(in, out, mag, n + 63);
}

void exec_mov_avg(cuFloatComplex* in,
                  float* mag,
                  cuFloatComplex* out,
                  float* cor,
                  int n,
                  int grid_size,
                  int block_size,
                  cudaStream_t stream)
{
    mov_avg_cplx_kernel<<<grid_size, block_size, 0, stream>>>(in, mag, out, cor, n);
}

void get_block_and_grid(int* minGrid, int* minBlock)
{
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, mov_avg_cplx_kernel, 0, 0);
}

}
} // namespace wifi
} // namespace gr