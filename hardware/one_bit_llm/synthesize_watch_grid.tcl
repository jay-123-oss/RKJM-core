# Synopsys Design Compiler / Vivado synthesis and PPA reporting flow.
#
# DC:
#   dc_shell -f synthesize_watch_grid.tcl
# Vivado:
#   vivado -mode batch -source synthesize_watch_grid.tcl \
#     -tclargs vivado xc7z020clg400-1
#
# The 1.5 GHz target implies a 0.666667 ns clock period.  Technology-specific
# .db/.lib files and a real standard-cell/FPGA part must be supplied by the
# invoking environment; expected PPA values below are planning estimates, not
# fabricated tool reports.

set TOP watch_grid_4x4
set RTL_FILE watch_grid_4x4.sv
set CLOCK_NAME clk
set CLOCK_PERIOD_NS 0.666667
set CLOCK_UNCERTAINTY_NS 0.033333
set OUTPUT_DIR reports_watch_grid
file mkdir $OUTPUT_DIR
if {[info exists ::env(TARGET_LIBRARY)]} {
    set TARGET_LIBRARY $::env(TARGET_LIBRARY)
} else {
    set TARGET_LIBRARY ""
}

proc report_expected_ppa {} {
    puts "Expected planning PPA (4x4, 1.5 GHz target; not measured silicon):"
    puts "  Static power : 0.10-0.35 mW (7nm), 0.25-0.80 mW (12nm)"
    puts "  Dynamic power: 1.5-4.5  mW (7nm), 3.0-9.0  mW (12nm)"
    puts "  Area         : 0.002-0.008 mm^2 (7nm), 0.006-0.020 mm^2 (12nm)"
    puts "  Final values require characterized libraries, parasitics, activity, and IR-drop analysis."
}

if {[llength [info commands analyze]] > 0} {
    # Synopsys Design Compiler flow.
    if {$TARGET_LIBRARY eq ""} {
        error "Set TARGET_LIBRARY to a characterized .db/.lib before running DC."
    }
    set_app_var search_path [list .]
    set_app_var target_library [list "${TARGET_LIBRARY}" "*"]
    set_app_var link_library [list "*" "${TARGET_LIBRARY}"]
    analyze -format sverilog $RTL_FILE
    elaborate $TOP
    current_design $TOP
    link

    create_clock -name $CLOCK_NAME -period $CLOCK_PERIOD_NS [get_ports clk]
    set_clock_uncertainty $CLOCK_UNCERTAINTY_NS [get_clocks $CLOCK_NAME]
    set_input_delay 0.10 -clock $CLOCK_NAME [remove_from_collection [all_inputs] [get_ports clk]]
    set_output_delay 0.10 -clock $CLOCK_NAME [all_outputs]
    set_false_path -from [get_ports reset_n]

    # FW_BW_SELECT isolates the inactive registered domain.  This directive
    # permits DC to infer integrated clock gating where the library supports it.
    set_clock_gating_style -positive_edge_logic integrated \
        -control_point before -control_signal FW_BW_SELECT
    set_clock_gating_check -setup 0.05 -hold 0.02 [get_clocks $CLOCK_NAME]
    set_dont_touch_network [get_ports reset_n]

    compile_ultra -gate_clock
    write -format verilog -hierarchy -output "${OUTPUT_DIR}/${TOP}_gate.v"
    write_sdc "${OUTPUT_DIR}/${TOP}.sdc"
    report_timing -max_paths 20 > "${OUTPUT_DIR}/timing.rpt"
    report_area -hierarchy > "${OUTPUT_DIR}/area.rpt"
    report_power -hierarchy > "${OUTPUT_DIR}/power.rpt"
    report_qor > "${OUTPUT_DIR}/qor.rpt"
    report_expected_ppa
} elseif {[llength [info commands read_verilog]] > 0} {
    # Vivado flow.  Use a real device part supplied as the second tcl argument.
    set part [lindex $::argv 1]
    if {$part eq ""} { error "Vivado requires a part: -tclargs vivado <part>" }
    read_verilog -sv $RTL_FILE
    synth_design -top $TOP -part $part -flatten_hierarchy rebuilt
    create_clock -name $CLOCK_NAME -period $CLOCK_PERIOD_NS [get_ports clk]
    set_clock_uncertainty $CLOCK_UNCERTAINTY_NS [get_clocks $CLOCK_NAME]
    set_false_path -from [get_ports reset_n]
    opt_design
    place_design
    route_design
    report_timing_summary -file "${OUTPUT_DIR}/timing.rpt"
    report_utilization -hierarchical -file "${OUTPUT_DIR}/area.rpt"
    report_power -file "${OUTPUT_DIR}/power.rpt"
    write_verilog -force "${OUTPUT_DIR}/${TOP}_gate.v"
    report_expected_ppa
} else {
    puts "No DC or Vivado commands detected. Review constraints and install a synthesis tool."
    report_expected_ppa
}