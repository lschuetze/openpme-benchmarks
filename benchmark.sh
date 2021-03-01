#!/bin/bash
set -e
pushd `dirname $0` > /dev/null
SCRIPT_PATH=`pwd`
popd > /dev/null
USER_BASE_PATH=`python3 -m site --user-base`
# Script start
$USER_BASE_PATH/bin/rebench --clean $SCRIPT_PATH/bench.conf s:openfpm