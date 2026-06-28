#!/bin/bash
script_path=$(realpath $(dirname $0))
echo ASCEND_HOME_PATH=${ASCEND_HOME_PATH}

mkdir -p build_out
rm -rf build_out/*
cd build_out

cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')
if [ "$cmake_version" \< "3.19.0" ] ; then
    opts=$(python3 $script_path/cmake/util/preset_parse.py $script_path/CMakePresets.json)
    echo $opts
    cmake .. $opts
else
    cmake .. --preset=default -DASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH
fi
target=package
if [ "$1"x != ""x ]; then target=$1; fi

cmake --build . --target $target -j16
if [ $? -ne 0 ]; then exit 1; fi

if [ $target = "package" ]; then
  if test -d ./op_kernel/binary ; then
    ./cust*.run
    if [ $? -ne 0 ]; then exit 1; fi
    cmake --build . --target binary -j16
    if [ $? -ne 0 ]; then exit 1; fi
    cmake --build . --target $target -j16
  fi
fi

# for debug
# cd build_out
# make
# cpack
# verbose append -v
