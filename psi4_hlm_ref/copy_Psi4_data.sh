#!/bin/bash

sim_list=(
# IRE3.0_ALF2_ALF2_030_M2.57_hydro_only
# IRE3.0_ALF2_ALF2_030_M2.57_B1.38E16
# IRE3.0_ALF2_ALF2_030_M2.57_B2.20E17
# IRE3.0_ALF2_ALF2_030_M2.57_B5.50E15
# IRE3.0_ALF2_ALF2_030_M2.57_B5.50E16
# IRE3.0_ALF2_ALF2_030_M2.57_pol_int_B5.50E16
# IRE3.0_ALF2_ALF2_030_M2.57_tor_int_B5.50E16
# IRE3.0_ALF2_ALF2_030_M2.70_hydro_only
# IRE3.0_ALF2_ALF2_030_M2.70_B1.38E16
# IRE3.0_ALF2_ALF2_030_M2.70_B2.20E17
# IRE3.0_ALF2_ALF2_030_M2.70_B5.50E15
# IRE3.0_ALF2_ALF2_030_M2.70_B5.50E16
# IRE3.0_ALF2_ALF2_030_M2.70_pol_int_B5.50E16
# IRE3.0_ALF2_ALF2_030_M2.70_tor_int_B5.50E16
# IRE3.0_SLy_SLy_010_M2.57_hydro_only
# IRE3.0_SLy_SLy_010_M2.57_B1.38E16
# IRE3.0_SLy_SLy_010_M2.57_B2.20E17
# IRE3.0_SLy_SLy_010_M2.57_B5.50E15
# IRE3.0_SLy_SLy_010_M2.57_B5.50E16
# IRE3.0_SLy_SLy_010_M2.57_pol_int_B5.50E16
# IRE3.0_SLy_SLy_010_M2.57_tor_int_B5.50E16
IRE3.0_ALF2_ALF2_030_M2.70_B1.38E16
IRE3.0_ALF2_ALF2_030_M2.70_B2.20E17
IRE3.0_ALF2_ALF2_030_M2.70_B5.50E15
IRE3.0_ALF2_ALF2_030_M2.70_B5.50E16
IRE3.0_ALF2_ALF2_030_M2.70_hydro_only
IRE3.0_ALF2_ALF2_030_M2.70_pol_int_B5.50E16
IRE3.0_ALF2_ALF2_030_M2.70_tor_int_B5.50E16
)

out_dir=NSNS_2025_Psi_4_data
data_dir=/data/shared/Jamie_f2_Effect_of_Bfield_on_NSNS
other_data_dir=/data/jbamber/Effect_of_Bfield_on_NSNS_data

for sim in "${sim_list[@]}"; do
  for Psi_num in 3 # 4 5 6 7
  do
    #diff $data_dir/$sim/data/Psi4_rad.mon_sorted.$Psi_num $other_data_dir/$sim/data/Psi4_rad.mon_sorted.$Psi_num
    #tail -n 1 $data_dir/$sim/data/ejv_GW.${Psi_num}.dat 
    #
    #tail -n 1 $other_data_dir/$sim/data/ejv_GW.${Psi_num}.dat 
    echo $data_dir " version"
    tail -n 1 $data_dir/$sim/data/ejv_GW.${Psi_num}.dat
    echo $other_data_dir " version"
    tail -n 1 $other_data_dir/$sim/data/ejv_GW.${Psi_num}.dat
    echo ""
    echo ""
  done
done

