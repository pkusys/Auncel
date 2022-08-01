#!/bin/sh

# multipler = nprobe/100

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db SIFT10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=7530 -k 100 -err 10 > result/result_SIFT10M_IVF1024_Flat_tree6_b1

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db DEEP10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=8910 -k 100 -err 10 > result/result_DEEP10M_IVF1024_Flat_tree6_b1

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 3 -db GIST1M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=11210 -k 100 -err 10 > result/result_GIST1M_IVF1024_Flat_tree6_b1

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db TEXT10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=23110 -k 100 -err 10 > result/result_TEXT10M_IVF1024_Flat_tree6_b1

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db SIFT10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=4250 -k 50 -err 10 > result/result_SIFT10M_IVF1024_Flat_tree6_b1

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db SIFT10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=15030 -k 10 -err 10 > result/result_SIFT10M_IVF1024_Flat_tree6_b1

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db SIFT10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=15260 -k 100 -err 5 > result/result_SIFT10M_IVF1024_Flat_tree6_b1

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db SIFT10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=48800 -k 100 -err 1 > result/result_SIFT10M_IVF1024_Flat_tree6_b1


# sudo rm /data/zzl/workspace/faiss-learned-termination/benchs/learned_termination/trained_index/
# sudo rm /data/zzl/workspace/faiss-learned-termination/benchs/learned_termination/populated_index/
# sudo rm /data/zzl/workspace/faiss-learned-termination/benchs/learned_termination/ground_truth/

