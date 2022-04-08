/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "sync_long_cuda.cuh"

// The block cuda file is just a wrapper for the kernels that will be launched in the work
// function
namespace gr {
namespace wifi {
namespace sync_long_cu {
__global__ void apply_sync_long_kernel(const uint8_t* in, uint8_t* out, int batch_size)
{
    // block specific code goes here
}

void apply_sync_long(
    const uint8_t* in, uint8_t* out, int grid_size, int block_size, cudaStream_t stream)
{
    int batch_size = block_size * grid_size;
    apply_sync_long_kernel<<<grid_size, block_size, 0, stream>>>(in, out, batch_size);
}

void get_block_and_grid(int* minGrid, int* minBlock)
{
    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, apply_sync_long_kernel, 0, 0);
}

__global__ void
remove_cp(cuFloatComplex* in, cuFloatComplex* out, int symlen, int cplen, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int sym_idx = i / symlen;
        int samp_idx = i % symlen;

        if (samp_idx >= cplen) {
            out[sym_idx * (symlen - cplen) + samp_idx - cplen] =
                in[sym_idx * symlen + samp_idx];
        }
    }
}

__global__ void remove_cp_freqcorr(cuFloatComplex* in,
                                   cuFloatComplex* out,
                                   int symlen,
                                   int cplen,
                                   int n,
                                   float freqoff,
                                   int start_sym)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int sym_idx = i / symlen;
        int samp_idx = i % symlen;

        if (samp_idx >= cplen) {
            out[sym_idx * (symlen - cplen) + samp_idx - cplen] =
                in[sym_idx * symlen + samp_idx];
            float x = ((start_sym + sym_idx) * symlen + samp_idx) * freqoff;
            cuCmulf(in[i], make_cuFloatComplex(cos(x), sin(x)));
        }
    }
}

void exec_remove_cp(cuFloatComplex* in,
                    cuFloatComplex* out,
                    int symlen,
                    int cplen,
                    int n,
                    int grid_size,
                    int block_size,
                    cudaStream_t stream)
{
    remove_cp<<<grid_size, block_size, 0, stream>>>(in, out, symlen, cplen, n);
}

void exec_remove_cp_freqcorr(cuFloatComplex* in,
                             cuFloatComplex* out,
                             int symlen,
                             int cplen,
                             int n,
                             int grid_size,
                             int block_size,
                             float freqoff,
                             int start_sym,
                             cudaStream_t stream)
{
    remove_cp_freqcorr<<<grid_size, block_size, 0, stream>>>(
        in, out, symlen, cplen, n, freqoff, start_sym);
}

__global__ void
multiply_kernel_ccc(cuFloatComplex* in1, cuFloatComplex* in2, cuFloatComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float re, im;
        re = in1[i].x * in2[i].x - in1[i].y * in2[i].y;
        im = in1[i].x * in2[i].y + in1[i].y * in2[i].x;
        out[i].x = re;
        out[i].y = im;
    }
}

void exec_multiply_kernel_ccc(cuFloatComplex* in1,
                              cuFloatComplex* in2,
                              cuFloatComplex* out,
                              int n,
                              int grid_size,
                              int block_size,
                              cudaStream_t stream)
{
    multiply_kernel_ccc<<<grid_size, block_size, 0, stream>>>(in1, in2, out, n);
}


} // namespace sync_long_cu
} // namespace wifi
} // namespace gr