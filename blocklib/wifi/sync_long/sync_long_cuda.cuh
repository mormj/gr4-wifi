/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */
#pragma once

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
namespace wifi {
namespace sync_long_cu {

void apply_sync_long(
    const uint8_t* in, uint8_t* out, int grid_size, int block_size, cudaStream_t stream);

void get_block_and_grid(int* minGrid, int* minBlock);

void exec_remove_cp(cuFloatComplex* in,
                    cuFloatComplex* out,
                    int symlen,
                    int cplen,
                    int n,
                    int grid_size,
                    int block_size,
                    cudaStream_t stream);

void exec_remove_cp_freqcorr(cuFloatComplex* in,
                             cuFloatComplex* out,
                             int symlen,
                             int cplen,
                             int n,
                             int grid_size,
                             int block_size,
                             float freqoff,
                             int start_sym,
                             cudaStream_t stream);

void exec_multiply_kernel_ccc(cuFloatComplex* in1,
                              cuFloatComplex* in2,
                              cuFloatComplex* out,
                              int n,
                              int grid_size,
                              int block_size,
                              cudaStream_t stream);
} // namespace sync_long_cu
} // namespace wifi
} // namespace gr