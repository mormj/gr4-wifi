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
namespace sync_short_cu {

template <typename T>
void exec_kernel(
    const T* in, T* out, int grid_size, int block_size, cudaStream_t stream);

void get_block_and_grid(int* minGrid, int* minBlock);

} // namespace sync_short
} // namespace wifi
} // namespace gr