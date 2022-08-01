#!/bin/sh

make clean
cd python/
make clean
cd ..
make -j4
make install
make py

