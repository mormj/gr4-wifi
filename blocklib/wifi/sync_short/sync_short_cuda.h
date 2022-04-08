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

#include <gnuradio/wifi/sync_short.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
namespace wifi {

class sync_short_cuda : public sync_short
{
public:
    sync_short_cuda(block_args args);
    virtual work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

private:
    void insert_tag(uint64_t item,
                    double freq_offset,
                    uint64_t input_item,
                    block_work_output_sptr& work_output)
    {
        work_output->add_tag(item,
                                { { "wifi_start", freq_offset }, { "srcid", name() } });
    }

    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;

    int d_min_plateau;
    float d_threshold;
    uint64_t d_last_tag_location = 0;
    float d_freq_offset;

    size_t packet_cnt = 0;
	
    cuFloatComplex* d_dev_in;
    cuFloatComplex* d_dev_out;

    std::vector<uint8_t> above_threshold;
    std::vector<uint8_t> accum;
	std::vector<float> d_host_cor;
	std::vector<gr_complex> d_host_abs;

    static const int MIN_GAP = 480;
    static const int MAX_SAMPLES = 540 * 80;

    static const int d_max_out_buffer = 1024*1024*2; // max bytes for output buffer
};

} // namespace wifi
} // namespace gr