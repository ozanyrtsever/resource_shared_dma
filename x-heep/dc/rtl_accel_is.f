# Filelist for the CURRENT accelerator (batched GEMM), LOGIC-ONLY area.
# tb/xbuf_ram.sv is DELIBERATELY OMITTED so the u_xbuf input buffer links as a
# black box (area 0) -> the reported area is the accelerator's control/datapath
# logic only. The buffer is a 32 KB single-port SRAM macro, reported separately.
# (Expect harmless "unresolved reference xbuf_ram" warnings during link.)
#
# FALLBACK (if your DC refuses to black-box a missing module): add tb/xbuf_ram.sv
# below, then in synth_area.tcl add `set_dont_touch [get_cells u_xbuf]` before
# compile and read the u_xbuf row's area from area_*_hier.rpt (LOGIC = total - u_xbuf).
hw/core-v-mini-mcu/include/fifo_pkg.sv
tb/dma_fp_dot_accel_is.sv
