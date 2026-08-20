# Filelist for the PIPELINED accelerator (batched GEMM, pipelined issue), LOGIC-ONLY area.
# Same convention as rtl_accel_is.f: tb/xbuf_ram.sv is DELIBERATELY OMITTED so the u_xbuf input
# buffer links as a black box (area 0) -> the reported area is the accelerator's control/datapath
# logic only. The buffer is a 32 KB single-port SRAM macro, reported separately.
# (Expect harmless "unresolved reference xbuf_ram" warnings during link.)
hw/core-v-mini-mcu/include/fifo_pkg.sv
tb/dma_fp_dot_accel_pipe.sv
