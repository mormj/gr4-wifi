/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "pre_sync_cuda.h"
#include "pre_sync_cuda_gen.h"

#include <gnuradio/helper_cuda.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace gr {
namespace wifi {

pre_sync_cuda::pre_sync_cuda(block_args args)
    : INHERITED_CONSTRUCTORS,
      d_buffer_size(args.buffer_size),
      d_window_size(args.window_size)

{
    pre_sync_cu::get_block_and_grid(&d_min_grid_size, &d_block_size);
    d_logger->info("minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);

    checkCudaErrors(cudaMalloc((void**)&dev_buf_1, d_buffer_size));
    checkCudaErrors(cudaMalloc((void**)&dev_buf_2, d_buffer_size));

    cudaStreamCreate(&d_stream);
}

work_return_code_t pre_sync_cuda::work(std::vector<block_work_input_sptr>& work_input,
                                       std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<gr_complex>();
    auto out = work_output[0]->items<gr_complex>();
    auto abs = work_output[1]->items<gr_complex>();
    auto cor = work_output[2]->items<float>();

    auto hist_samps = d_window_size + 16 - 1;

    // input buffer needs to be > noutput_items + hist_samps
    int noutput = std::min(std::min(work_output[0]->n_items, work_output[1]->n_items),
                           work_output[2]->n_items);
    int ninput = work_input[0]->n_items;

    // adjust the inputs and outputs
    if (ninput < (int)(noutput + hist_samps + 16)) {
        noutput = ninput - hist_samps - 16;
    } else {
        ninput = noutput + hist_samps + 16;
    }


    auto gridSize = (noutput + 1024 - 1) / 1024;
    pre_sync_cu::exec_corr_abs(((cuFloatComplex*)in),
                               (cuFloatComplex*)dev_buf_1,
                               (float*)dev_buf_2,
                               noutput,
                               gridSize,
                               1024,
                               d_stream);

    pre_sync_cu::exec_mov_avg((cuFloatComplex*)dev_buf_1,
                              (float*)dev_buf_2,
                              (cuFloatComplex*)abs,
                              cor,
                              noutput,
                              gridSize,
                              1024,
                              d_stream);

    // memcpy(out, in + 47, noutput_items * sizeof(gr_complex));
    checkCudaErrors(cudaMemcpyAsync(
        out, in + 47, sizeof(gr_complex) * noutput, cudaMemcpyDeviceToDevice, d_stream));

    cudaStreamSynchronize(d_stream);

    // add_item_tag(1, nitems_written(0), pmt::mp("frame"), pmt::from_long(0));

    consume_each(noutput, work_input);
    produce_each(noutput, work_output);
    return work_return_code_t::WORK_OK;
}
} // namespace wifi
} // namespace gr
