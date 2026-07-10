read netlist /data4/pengqingsong/DFT/TPI-my.3/autoresearch/eval8-real-priors-budget5-v1/fault_logs/benchmarks/epfl__random_control__i2c__i2c/baseline/tmax_logs/netlist_full_scan.v
set build -nodelete_unused_gates -merge noxor_from_gates -merge noequivalent_dlat_dff -merge noflipflop_from_dlat -merge nofeedback_paths -merge nomux_from_gates -merge notied_gates_with_pin_loss -merge nowire_to_buffer -merge nocascaded_gates_with_pin_loss
run build_model tpi_top
run drc
set faults -model stuck -report uncollapsed -fault_coverage
add faults -all
set patterns -delete
set random_patterns -length 300000
set patterns -random
run fault_sim
report summaries > /data4/pengqingsong/DFT/TPI-my.3/autoresearch/eval8-real-priors-budget5-v1/fault_logs/benchmarks/epfl__random_control__i2c__i2c/baseline/tmax_logs/atpg_summary.rpt
report patterns -summary >> /data4/pengqingsong/DFT/TPI-my.3/autoresearch/eval8-real-priors-budget5-v1/fault_logs/benchmarks/epfl__random_control__i2c__i2c/baseline/tmax_logs/atpg_summary.rpt
report faults -all > /data4/pengqingsong/DFT/TPI-my.3/autoresearch/eval8-real-priors-budget5-v1/fault_logs/benchmarks/epfl__random_control__i2c__i2c/baseline/tmax_logs/faults.rpt
exit -force
