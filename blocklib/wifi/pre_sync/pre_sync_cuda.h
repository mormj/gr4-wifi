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

#include <gnuradio/wifi/pre_sync.h>

#include "pre_sync_cuda.cuh"

namespace gr {
namespace wifi {

class pre_sync_cuda : public pre_sync
{
public:
    pre_sync_cuda(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    size_t d_buffer_size;
    size_t d_window_size;

    void* dev_buf_1;
    void* dev_buf_2;

    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};

} // namespace wifi
} // namespace gr