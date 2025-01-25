#!/bin/sh

cd $PBS_O_WORKDIR

python3.8 feature_computation.py $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8
