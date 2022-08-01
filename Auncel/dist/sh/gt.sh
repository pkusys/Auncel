#!/bin/sh
../gt /workspace/data/dist/NO4/0db.fbin >../logs/NO4/0db-gt.log 
../gt /workspace/data/dist/NO4/1db.fbin >../logs/NO4/1db-gt.log

# kill -9 $(pidof gt)