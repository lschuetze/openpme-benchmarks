#!/bin/bash

# CORES $1, SUITE $2, BENCHMARK $3

case $3 in 
    LennardJones)

    ;;

    GrayScott)
    mpirun -np $1 ../benchmarks/$2/2_gray_scott/gray_scott
    ;;

    VortexInCell)

    ;;
esac