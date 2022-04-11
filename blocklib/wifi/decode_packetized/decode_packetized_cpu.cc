/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "decode_packetized_cpu.h"
#include "decode_packetized_cpu_gen.h"

#include "utils.h"
#include "viterbi_decoder/viterbi_decoder.h"
#include <boost/crc.hpp>

#include <pmtf/map.hpp>

namespace gr {
namespace wifi {

decode_packetized_cpu::decode_packetized_cpu(block_args args)
    : INHERITED_CONSTRUCTORS, d_log(args.log), d_debug(args.debug), d_frame(d_ofdm, 0)
{

    d_equalizer = new equalizer::ls();

    d_bpsk = constellation_bpsk::make();
    d_qpsk = constellation_qpsk::make();
    d_16qam = constellation_16qam::make();
    d_64qam = constellation_64qam::make();
}

void decode_packetized_cpu::handle_msg_pdus(pmtf::pmt msg)
{
    // std::cout << "got msg" << std::endl;
    static int nrx = 0;
                    nrx++;

                    if (nrx % 1000 == 0) {
                        std::cout << "decode (rx): " << nrx << std::endl;
                    }
    auto meta = pmtf::get_as<std::map<std::string, pmtf::pmt>>(pmtf::get_as<std::map<std::string, pmtf::pmt>>(msg)["meta"]);
    auto samples = pmtf::get_as<std::vector<gr_complex>>(pmtf::get_as<std::map<std::string, pmtf::pmt>>(msg)["data"]);

    d_frame_bytes = pmtf::get_as<int>(meta["frame_bytes"]);
    d_frame_symbols = pmtf::get_as<int>(meta["frame_symbols"]);
    d_frame_encoding = pmtf::get_as<int>(meta["encoding"]);
    d_bw = pmtf::get_as<double>(meta["bw"]);
    d_freq = pmtf::get_as<double>(meta["freq"]);
    d_freq_offset = pmtf::get_as<double>(meta["freq_offset"]);

    switch (d_frame_encoding) {
    case 0:
    case 1:
        d_frame_mod = d_bpsk;
        break;
    case 2:
    case 3:
        d_frame_mod = d_qpsk;
        break;
    case 4:
    case 5:
        d_frame_mod = d_16qam;
        break;
    case 6:
    case 7:
        d_frame_mod = d_64qam;
        break;
    default:
        throw new std::runtime_error("invalid encoding");
    }

    auto H = pmtf::get_as<std::vector<gr_complex>>(meta["H"]);
    auto tmp_prev_pilots = pmtf::get_as<std::vector<gr_complex>>(meta["prev_pilots"]);
    memcpy(d_prev_pilots, tmp_prev_pilots.data(), 4 * sizeof(gr_complex));
    d_equalizer->set_H(H.data());

    equalize_frame(samples.data(), d_rx_symbols);


    d_ofdm = ofdm_param((Encoding)d_frame_encoding);
    d_frame = frame_param(d_ofdm, d_frame_bytes);

    // need to equalize and demap the samples to d_rx_bits
    if (!decode(d_rx_bits,
                d_rx_symbols,
                d_deinterleaved_bits,
                out_bytes,
                d_decoder,
                d_frame,
                d_ofdm)) {
        // FILE *pFile;
        // char tmp[1024];
        // sprintf(tmp,"/tmp/decode_%d.dat", this->id());
        // pFile = fopen(tmp, "a");
        // fprintf(pFile, "x,%d,%d,%d,%d,%.1f,%.1f,%.6f,%d,", pc, d_frame_encoding,
        // d_frame_bytes, d_frame_symbols, d_bw, d_freq, d_freq_offset, len_bytes);
        // // for (int i=0; i<64*23; i++)
        // // {
        // //     fprintf(pFile, "%.6f+%.6f,", real(samples[i]), imag(samples[i]));
        // // }
        // for (int i=0; i<64; i++)
        // {
        //     fprintf(pFile, "%.6f+%.6f,", real(H[i]), imag(H[i]));
        // }
        // for (int i=0; i<4; i++)
        // {
        //     fprintf(pFile, "%.6f+%.6f,", real(d_prev_pilots[i]),
        //     imag(d_prev_pilots[i]));
        // }
        // fprintf(pFile, "\n");
        // for (int i=0; i<d_frame_symbols*64; i++)
        // {
        //     fprintf(pFile, "%.6f+%.6f,", real(samples[i]), imag(samples[i]));
        // }
        // fprintf(pFile, "\n");
        // for (int i=0; i<d_frame.n_sym*48; i++)
        // {
        //     fprintf(pFile, "%d,", d_rx_symbols[i]);
        // }
        // fprintf(pFile, "\n");

        // // fwrite(rx_bits, 1, frame_info.n_sym * 48 , pFile);
        // fclose(pFile);
    } else {

        // FILE *pFile;
        // char tmp[1024];
        // sprintf(tmp,"/tmp/decode_%d.dat", this->id());
        // pFile = fopen(tmp, "a");
        // fprintf(pFile, "o,%d,%d,%d,%d,%.1f,%.1f,%.6f,%d,", pc, d_frame_encoding,
        // d_frame_bytes, d_frame_symbols, d_bw, d_freq, d_freq_offset, len_bytes);
        // // for (int i=0; i<64*23; i++)
        // // {
        // //     fprintf(pFile, "%.6f+%.6f,", real(samples[i]), imag(samples[i]));
        // // }
        // for (int i=0; i<64; i++)
        // {
        //     fprintf(pFile, "%.6f+%.6f,", real(H[i]), imag(H[i]));
        // }
        // for (int i=0; i<4; i++)
        // {
        //     fprintf(pFile, "%.6f+%.6f,", real(d_prev_pilots[i]),
        //     imag(d_prev_pilots[i]));
        // }
        // fprintf(pFile, "\n");
        // for (int i=0; i<d_frame_symbols*64; i++)
        // {
        //     fprintf(pFile, "%.6f+%.6f,", real(samples[i]), imag(samples[i]));
        // }
        // fprintf(pFile, "\n");
        // for (int i=0; i<d_frame.n_sym*48; i++)
        // {
        //     fprintf(pFile, "%d,", d_rx_symbols[i]);
        // }
        // fprintf(pFile, "\n");

        // // fwrite(rx_bits, 1, frame_info.n_sym * 48 , pFile);
        // fclose(pFile);
    }

    // Insert MAC Decode code here
    // std::cout << "Threadpool got new burst" << std::endl;

    this->packet_cnt++;
    // if (packet_cnt % 100 == 0)
        // std::cout << "decoded: " << packet_cnt << std::endl;

    // send the pdu out the output port

    // return pdu;
}

void decode_packetized_cpu::deinterleave()
{

    int n_cbps = d_ofdm.n_cbps;
    int first[n_cbps];
    int second[n_cbps];
    int s = std::max(d_ofdm.n_bpsc / 2, 1);

    for (int j = 0; j < n_cbps; j++) {
        first[j] = s * (j / s) + ((j + int(floor(16.0 * j / n_cbps))) % s);
    }

    for (int i = 0; i < n_cbps; i++) {
        second[i] = 16 * i - (n_cbps - 1) * int(floor(16.0 * i / n_cbps));
    }

    for (int i = 0; i < d_frame.n_sym; i++) {
        for (int k = 0; k < n_cbps; k++) {
            d_deinterleaved_bits[i * n_cbps + second[first[k]]] =
                d_rx_bits[i * n_cbps + k];
        }
    }
}


void decode_packetized_cpu::descramble(uint8_t* decoded_bits)
{

    int state = 0;
    std::memset(out_bytes, 0, d_frame.psdu_size + 2);

    for (int i = 0; i < 7; i++) {
        if (decoded_bits[i]) {
            state |= 1 << (6 - i);
        }
    }
    out_bytes[0] = state;

    int feedback;
    int bit;

    for (int i = 7; i < d_frame.psdu_size * 8 + 16; i++) {
        feedback = ((!!(state & 64))) ^ (!!(state & 8));
        bit = feedback ^ (decoded_bits[i] & 0x1);
        out_bytes[i / 8] |= bit << (i % 8);
        state = ((state << 1) & 0x7e) | feedback;
    }
}

void decode_packetized_cpu::print_output()
{

    dout << std::endl;
    dout << "psdu size" << d_frame.psdu_size << std::endl;
    for (int i = 2; i < d_frame.psdu_size + 2; i++) {
        dout << std::setfill('0') << std::setw(2) << std::hex
             << ((unsigned int)out_bytes[i] & 0xFF) << std::dec << " ";
        if (i % 16 == 15) {
            dout << std::endl;
        }
    }
    dout << std::endl;
    for (int i = 2; i < d_frame.psdu_size + 2; i++) {
        if ((out_bytes[i] > 31) && (out_bytes[i] < 127)) {
            dout << ((char)out_bytes[i]);
        } else {
            dout << ".";
        }
    }
    dout << std::endl;
}

} // namespace wifi
} // namespace gr
