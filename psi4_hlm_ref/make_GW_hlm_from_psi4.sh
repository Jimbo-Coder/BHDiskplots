#!/bin/bash

# Script to use the scripts in psi4_hlm to generate the h_lm and GW fluxes from the Psi4 outputs 
# from the NS-NS simulations 

home_dir=$(pwd)

sim_names=(
bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_28
bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_32
bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_35
bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_01
bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_05
bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_07
)

declare -A omega_dict=(
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_28"]=0.363729746185521
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_32"]=0.305581353575123
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_35"]=0.257573177238334
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_01"]=0.385761187900437
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_05"]=0.342298350854386
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_07"]=0.299915740314243
)

declare -A m_adm_dict=(
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_28"]=0.0508101490724165
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_32"]=0.0564177477296656
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_35"]=0.0840478939414561
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_01"]=0.0512207666580524
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_05"]=0.0603349020955639
["bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_07"]=0.0807943547824248
)

scratch_folder=/data/jbamber/BH_massiveDisk

for sim_name in ${sim_names[@]}
do
  cd $scratch_folder/$sim_name/data
  cp Psi4_rad.mon.\* Psi4_rad.mon.10
  for Psi4_file_num in 1 2 3 4 5 6 7 8 9 10
  do

    cat Psi4_rad.mon.${Psi4_file_num} > Psi4_rad.mon.input_file
    #tail -n +2
    cat Psi4_rad.mon.input_file | sort -k1 -g -u | tail -n +2 > Psi4_rad.mon_sorted.${Psi4_file_num}

    # if [ "${full_sim_name}" == "IRE3.0_ALF2_ALF2_030_M2.70_hydro_only" ] || [ "${full_sim_name}" == "IRE3.0_ALF2_ALF2_030_M2.57_hydro_only" ] || [ "${full_sim_name}" == "IRE3.0_SLy_SLy_010_M2.57_hydro_only" ] 
    # then
    #      echo "need to remove first line"
    #      tail -n +2 Psi4_rad.mon_sorted.${Psi4_file_num} > Psi4_rad.mon_sorted_2.${Psi4_file_num}
    #      cat Psi4_rad.mon_sorted_2.${Psi4_file_num} > Psi4_rad.mon_sorted.${Psi4_file_num}
    #      rm Psi4_rad.mon_sorted_2.${Psi4_file_num}
    # fi

    echo "Sorted Psi4_rad.mon.${Psi4_file_num}"

    head -n 1 Psi4_rad.mon_sorted.${Psi4_file_num} 

    initial_data_file=$scratch_folder/$hydro_only_name/bnsphyseq.dat

    omega_val=${omega_dict["$sim_name"]}
    m_adm_val=${m_adm_dict["$sim_name"]}
    t_start=-100
    t_end=100.0

    #declare $(awk '{if ($1=="Omega" && $2=="="){printf "omega_val=%.9g", $3}}' ${initial_data_file})
    #declare $(awk '{if ($1=="admmass_asymp" && $2=="="){printf "m_adm_val=%.9g", $3}}' ${initial_data_file})
    declare $(head -n1 Psi4_rad.mon_sorted.${Psi4_file_num} | awk '{printf "t_start=%.6g",$1}')
    declare $(tail -n1 Psi4_rad.mon_sorted.${Psi4_file_num} | awk '{printf "t_end=%.6g",$1}')

    echo "Omega = ${omega_val}"
    echo "M_ADM = ${m_adm_val}"
    echo "t_start = ${t_start}"
    echo "t_end = ${t_end}"
    t_end=`echo $t_end - 100.0 | bc`

    cp $home_dir/ccc_ffi.input_blank ccc_ffi.input

    sed -i "s|PSI4FNAME|Psi4_rad.mon_sorted.${Psi4_file_num}|" ccc_ffi.input
    sed -i "s|OMEGAVAL|${omega_val}|" ccc_ffi.input
    sed -i "s|ADMMASS|${m_adm_val}|" ccc_ffi.input
    sed -i "s|TSTART|${t_start}|" ccc_ffi.input
    sed -i "s|TEND|${t_end}|" ccc_ffi.input

    echo "We are in " $(pwd)

    $home_dir/rhphc

    cp $home_dir/gw_flux.input_blank gw_flux.input

    sed -i "s|ADMMASS|${m_adm_val}|" gw_flux.input
    sed -i "s|TSTARTVAL|${t_start}|" gw_flux.input
    sed -i "s|TENDVAL|${t_end}|" gw_flux.input

    #$home_dir/flux

    tail -n 1 ejv_GW.dat 
    mv ejv_GW.dat ejv_GW.${Psi4_file_num}.dat
    #mv EJ_rect.dat EJ_rect.${Psi4_file_num}.dat
    mv rhphc.dat rhphc.${Psi4_file_num}.dat
    mv rhphcdot.dat rhphcdot.${Psi4_file_num}.dat
    mv omega22.dat omega22.${Psi4_file_num}.dat
  done
done

cd $home_dir
