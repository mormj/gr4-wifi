/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "pre_sync_cpu.h"
#include "pre_sync_cpu_gen.h"

namespace gr {
namespace wifi {

pre_sync_cpu::pre_sync_cpu(block_args args) : INHERITED_CONSTRUCTORS {}

work_return_code_t pre_sync_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                      std::vector<block_work_output_sptr>& work_output)
{
    // Do <+signal processing+>
    // Block specific code goes here
    return work_return_code_t::WORK_OK;
}


} // namespace wifi
} // namespace gr