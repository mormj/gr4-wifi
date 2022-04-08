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

namespace gr {
namespace wifi {

class pre_sync_cpu : public virtual pre_sync
{
public:
    pre_sync_cpu(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    // private variables here
};

} // namespace wifi
} // namespace gr