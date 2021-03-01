#!/bin/bash

# CORES $1, SUITE $2, BENCHMARK $3

case $3 in 
    LennardJonesCL)
    mpirun -np $1 1_lennard_jones/md_dyn >/dev/null 2>&1
    ;;

    LennardJonesVL)
    mpirun -np $1 1_lennard_jones/md_dyn_vl >/dev/null 2>&1
    ;;

    GrayScott)
    mpirun -np $1 2_gray_scott/gray_scott >/dev/null 2>&1
    ;;

    VortexInCell)
    mpirun -np $1 3_vortex_in_cell/vic_petsc >/dev/null 2>&1
    ;;

    VortexInCellOpt)
    mpirun -np $1 3_vortex_in_cell/vic_petsc_opt >/dev/null 2>&1
    ;;
esac