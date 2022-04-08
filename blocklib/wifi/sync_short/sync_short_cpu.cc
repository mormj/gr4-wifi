/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "sync_short_cpu.h"
#include "sync_short_cpu_gen.h"

namespace gr {
namespace wifi {

sync_short_cpu::sync_short_cpu(block_args args) : INHERITED_CONSTRUCTORS {}

work_return_code_t sync_short_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                      std::vector<block_work_output_sptr>& work_output)
{
    // Do <+signal processing+>
    // Block specific code goes here
    return work_return_code_t::WORK_OK;
}


} // namespace wifi
} // namespace gr