#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: josh
# GNU Radio version: 0.2.0

from gnuradio import blocks
from gnuradio import fft
from gnuradio import fileio
from gnuradio import gr
#from gnuradio.filter import firdes
#from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
#from gnuradio.eng_arg import eng_float, intx
#from gnuradio import eng_notation
from gnuradio import wifi
import numpy as np
import time


def snipfcn_snippet_0(self):
    end = time.time()
    elapsed = end - self.startt
    #print(f"Received: {self.dbg.num_messages()} in {elapsed} seconds")
    print(f"Received: {self.dbg.num_messages()} in {elapsed} seconds")

def snipfcn_snippet_0_0(self):
    self.startt = time.time()


def snippets_main_after_init(fg):
    snipfcn_snippet_0_0(fg)

def snippets_main_after_stop(fg):
    snipfcn_snippet_0(fg)


class test_wifi(gr.flowgraph):

    def __init__(self):
        gr.flowgraph.__init__(self, "Not titled yet")

        ##################################################
        # Variables
        ##################################################
        self.sync_length = sync_length = 320
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################
        self.wifi_sync_short_0 = wifi.sync_short( .56,2,False,False, impl=wifi.sync_short.cuda)
        self.wifi_sync_long_0 = wifi.sync_long( sync_length,False,False, impl=wifi.sync_long.cuda)
        self.wifi_pre_sync_0 = wifi.pre_sync( 48,65536,False,False, impl=wifi.pre_sync.cuda)
        self.wifi_packetize_frame_0 = wifi.packetize_frame( 0,2462e6,20e6,False,False, impl=wifi.packetize_frame.cpu)
        self.fileio_file_source_0 = fileio.file_source( 8,'/data/data/cropcircles/wifi_synth_1500_1kpad_20MHz_10s_MCS0.fc32',False,0,0, impl=fileio.file_source.cpu)
        self.fft_fft_0 = fft.fft_cc_fwd( 64,np.ones(64),True, impl=fft.fft_cc_fwd.cuda)
        self.decode = wifi.decode_packetized( False,False, impl=wifi.decode_packetized.cpu)
        self.dbg = blocks.message_debug( False, impl=blocks.message_debug.cpu)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.fileio_file_source_0, 0), (self.wifi_pre_sync_0, 0)).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.H2D))
        self.connect((self.fft_fft_0, 0), (self.wifi_packetize_frame_0, 0)).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.D2H))
        self.connect((self.wifi_pre_sync_0, 0), (self.wifi_sync_short_0, 0)).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.D2D))
        self.connect((self.wifi_pre_sync_0, 2), (self.wifi_sync_short_0, 2)).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.D2D))
        self.connect((self.wifi_pre_sync_0, 1), (self.wifi_sync_short_0, 1)).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.D2D))
        self.connect((self.wifi_sync_long_0, 0), (self.fft_fft_0, 0)).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.D2D))
        self.connect((self.wifi_sync_short_0, 0), (self.wifi_sync_long_0, 0)).set_custom_buffer(gr.buffer_cuda_properties.make(gr.buffer_cuda_type.D2D))
        self.msg_connect((self.decode, 'out'), (self.dbg, 'store'))
        self.msg_connect((self.wifi_packetize_frame_0, 'pdus'), (self.decode, 'pdus'))


    def get_sync_length(self):
        return self.sync_length

    def set_sync_length(self, sync_length):
        self.sync_length = sync_length

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate




def main(flowgraph_cls=test_wifi, options=None):
    fg = flowgraph_cls()
    snippets_main_after_init(fg)
    def sig_handler(sig=None, frame=None):
        fg.stop()
        fg.wait()
        snippets_main_after_stop(fg)
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    fg.start()

    fg.wait()
    snippets_main_after_stop(fg)

if __name__ == '__main__':
    main()
