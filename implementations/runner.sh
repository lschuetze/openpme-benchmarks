#!/bin/bash

# CORES $1, SUITE $2, BENCHMARK $3

case $3 in 
    LennardJones)
    mpirun -np $1 1_lennard_jones/gray_scott
    ;;

    GrayScott)
    mpirun -np $1 2_gray_scott/gray_scott
    ;;

    VortexInCell)
    mpirun -np $1 3_vortex_in_cell/vic_petsc
    ;;

    VortexInCellOpt)
    mpirun -np $1 3_vortex_in_cell/vic_petsc_opt
    ;;
esac