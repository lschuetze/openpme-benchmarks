# OpenPME Benchmark Suite
The purpose is to compare the performance of [OpenPME](https://github.com/Nesrinekh/OpenPME) generated code to hand-written experiments written in [OpenFPM](http://openfpm.mpi-cbg.de). For statistics we use [Rebench](https://github.com/smarr/ReBench).

## Requirements
To run the benchmark the following requirements have to be fulfilled. Please make sure that all dependencies are installed:
- **OpenFPM**
    - Follow the instructions on their [website](http://openfpm.mpi-cbg.de/install#intro-wrapper)
    - After installing OpenFPM, copy the generated `example.mk` into the root folder of this benchmark. The file can be found in the root folder of the examples of your local OpenFPM installation.
- **ReBench**
    - `git submodule update --init`

## Usage
Execute `./benchmark.sh` in your terminal to run all benchmarks.