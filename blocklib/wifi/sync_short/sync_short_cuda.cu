/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

 #include "sync_short_cuda.cuh"

// The block cuda file is just a wrapper for the kernels that will be launched in the work
// function
namespace gr {
namespace wifi {
namespace sync_short_cu {
__global__ void apply_sync_short_kernel(const uint8_t* in, uint8_t* out, int batch_size)
{
    // block specific code goes here
}

void apply_sync_short(
    const uint8_t* in, uint8_t* out, int grid_size, int block_size, cudaStream_t stream)
{
    int batch_size = block_size * grid_size;
    apply_sync_short_kernel<<<grid_size, block_size, 0, stream>>>(in, out, batch_size);
}

void get_block_and_grid(int* minGrid, int* minBlock)
{
    // https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, apply_sync_short_kernel, 0, 0);
}
} // namespace sync_short_cu
} // namespace wifi
} // namespace gr