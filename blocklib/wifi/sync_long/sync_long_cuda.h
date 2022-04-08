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

#include <gnuradio/wifi/sync_long.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

namespace gr {
namespace wifi {

class sync_long_cuda : public sync_long
{
public:
    sync_long_cuda(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;

    enum { WAITING_FOR_TAG, FINISH_LAST_FRAME } d_state = WAITING_FOR_TAG;
    unsigned int d_sync_length;
    static const std::vector<gr_complex> LONG;
    int d_fftsize = 512;

    cufftHandle d_plan;
    cufftComplex* d_dev_training_freq;
    cufftComplex* d_dev_in;

    std::vector<gr::tag_t> tags;

    int d_ncopied = 0;
    float d_freq_offset = 0;
    float d_freq_offset_short = 0;

    int d_num_syms = 0;
    int ntags = 0;

    size_t packet_cnt = 0;

    int d_offset = 0;
};

} // namespace wifi
} // namespace gr