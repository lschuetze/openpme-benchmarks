#!/bin/bash
set -e
pushd `dirname $0` > /dev/null
SCRIPT_PATH=`pwd`
popd > /dev/null
# Script start
cd ./implementations/ReBench
python3 -m rebench.rebench --clean $SCRIPT_PATH/bench.conf s:openfpm:GrayScott