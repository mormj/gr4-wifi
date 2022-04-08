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

#include "equalizer/base.h"
#include "equalizer/comb.h"
#include "equalizer/lms.h"
#include "equalizer/ls.h"
#include "equalizer/sta.h"
#include "utils.h"
#include "viterbi_decoder/viterbi_decoder.h"
#include <gnuradio/wifi/constellations.h>
#include <gnuradio/wifi/frame_equalizer.h>
#include <gnuradio/wifi/wifi_types.h>

namespace gr {
namespace wifi {

class frame_equalizer_cpu : public virtual frame_equalizer
{
public:
    frame_equalizer_cpu(block_args args);
    virtual work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

private:
    void set_algorithm(Equalizer algo); // TODO replace with auto callback
    bool parse_signal(uint8_t* signal);
    bool decode_signal_field(uint8_t* rx_bits);
    void deinterleave(uint8_t* rx_bits);

    equalizer::base* d_equalizer = nullptr;
    std::mutex d_mutex;
    std::vector<gr::tag_t> tags;
    bool d_log;
    bool d_debug;
    int d_current_symbol = 0;
    viterbi_decoder d_decoder;

    // freq offset
    double d_freq;                            // Hz
    double d_freq_offset_from_synclong = 0.0; // Hz, estimation from "sync_long" block
    double d_bw;                              // Hz
    double d_er;
    double d_epsilon0;
    gr_complex d_prev_pilots[4];

    int d_frame_bytes = 0;
    int d_frame_symbols = 0;
    int d_frame_encoding;

    uint8_t d_deinterleaved[48];
    gr_complex symbols[48];

    std::shared_ptr<gr::kernel::digital::constellation> d_frame_mod;
    constellation_bpsk::sptr d_bpsk;
    constellation_qpsk::sptr d_qpsk;
    constellation_16qam::sptr d_16qam;
    constellation_64qam::sptr d_64qam;

    static const int interleaver_pattern[48];

    size_t ntags_rx = 0;
    size_t ntags_tx = 0;
};

} // namespace wifi
} // namespace gr