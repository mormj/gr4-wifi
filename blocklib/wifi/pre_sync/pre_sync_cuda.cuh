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
namespace pre_sync_cu {

void exec_corr_abs(cuFloatComplex* in,
                   cuFloatComplex* out,
                   float* mag,
                   int n,
                   int grid_size,
                   int block_size,
                   cudaStream_t stream);
void exec_mov_avg(cuFloatComplex* in,
                  float* mag,
                  cuFloatComplex* out,
                  float* cor,
                  int n,
                  int grid_size,
                  int block_size,
                  cudaStream_t stream);

void get_block_and_grid(int* minGrid, int* minBlock);

} // namespace pre_sync
} // namespace wifi
} // namespace gr