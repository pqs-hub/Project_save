read netlist /data4/pengqingsong/DFT/TPI-my.3/autoresearch/new-candidate-pool-insertion-smoke/evals/subckt_0001__beam__cumulative__reward_pred__bw4__d3__c256__g1p0__candrecall_pool__div0p02__s0/subckt_0001/baseline/tmax_logs/netlist_full_scan.v
set build -nodelete_unused_gates -merge noxor_from_gates -merge noequivalent_dlat_dff -merge noflipflop_from_dlat -merge nofeedback_paths -merge nomux_from_gates -merge notied_gates_with_pin_loss -merge nowire_to_buffer -merge nocascaded_gates_with_pin_loss
run build_model tpi_top
run drc
set faults -model stuck -report uncollapsed -fault_coverage
add faults -all
set patterns -delete
set random_patterns -length 10000
set patterns -random
run fault_sim
report summaries > /data4/pengqingsong/DFT/TPI-my.3/autoresearch/new-candidate-pool-insertion-smoke/evals/subckt_0001__beam__cumulative__reward_pred__bw4__d3__c256__g1p0__candrecall_pool__div0p02__s0/subckt_0001/baseline/tmax_logs/atpg_summary.rpt
report patterns -summary >> /data4/pengqingsong/DFT/TPI-my.3/autoresearch/new-candidate-pool-insertion-smoke/evals/subckt_0001__beam__cumulative__reward_pred__bw4__d3__c256__g1p0__candrecall_pool__div0p02__s0/subckt_0001/baseline/tmax_logs/atpg_summary.rpt
report faults -all > /data4/pengqingsong/DFT/TPI-my.3/autoresearch/new-candidate-pool-insertion-smoke/evals/subckt_0001__beam__cumulative__reward_pred__bw4__d3__c256__g1p0__candrecall_pool__div0p02__s0/subckt_0001/baseline/tmax_logs/faults.rpt
exit -force
