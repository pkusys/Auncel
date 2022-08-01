#!/bin/sh

make clean
cd eval
make clean
cd ..
make -j4
make install
cd eval
make