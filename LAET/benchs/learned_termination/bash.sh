# # test for end2end latency
# # python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thread 1 -thresh 2 -db DEEP10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=100 > result/result_DEEP10M_IVF1024_Flat_tree6_b1

# # python -u bench_learned_termination.py -mode 0 -batch 1 -cluster 1024 -db DEEP10M -idx IVF1024,Flat -param search_mode=0,nprobe=12 > result/result_DEEP10M_IVF4000_Flat_naive_b1

# # For deep10M
# python -u bench_learned_termination.py -mode -1 -batch 10000 -cluster 1024 -thread 10 -thresh 2 -db DEEP10M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_DEEP10M_IVF1024_Flat_test

# python -u bench_learned_termination.py -mode -2 -batch 100 -train 1 -cluster 1024 -thread 10 -thresh 2 -db DEEP10M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_DEEP10M_IVF1024_Flat_train

# python -u train_gbdt.py -train 1 -thresh 2 -db DEEP10M -idx IVF1024,Flat

# python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db DEEP10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=100 > result/result_DEEP10M_IVF1024_Flat_tree6_b1

# # For sift10M
# python -u bench_learned_termination.py -mode -1 -batch 10000 -cluster 1024 -thread 10 -thresh 2 -db SIFT10M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_SIFT10M_IVF1024_Flat_test

# python -u bench_learned_termination.py -mode -2 -batch 100 -train 1 -cluster 1024 -thread 10 -thresh 2 -db SIFT10M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_SIFT10M_IVF1024_Flat_train

# python -u train_gbdt.py -train 1 -thresh 2 -db SIFT10M -idx IVF1024,Flat

# python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db SIFT10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=100 > result/result_SIFT10M_IVF1024_Flat_tree6_b1

# # For gist1M
# python -u bench_learned_termination.py -mode -1 -batch 1 -cluster 1024 -thread 10 -thresh 3 -db GIST1M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_GIST1M_IVF1024_Flat_test

# python -u bench_learned_termination.py -mode -2 -batch 1 -train 1 -cluster 1024 -thread 10 -thresh 3 -db GIST1M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_GIST1M_IVF1024_Flat_train

# python -u train_gbdt.py -train 1 -thresh 3 -db GIST1M -idx IVF1024,Flat

# python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 3 -db GIST1M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=100 > result/result_GIST1M_IVF1024_Flat_tree6_b1

# # For spacev10M
# python -u bench_learned_termination.py -mode -1 -batch 10000 -cluster 1024 -thread 10 -thresh 2 -db SPACEV10M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_SPACEV10M_IVF1024_Flat_test

# python -u bench_learned_termination.py -mode -2 -batch 100 -train 1 -cluster 1024 -thread 10 -thresh 2 -db SPACEV10M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_SPACEV10M_IVF1024_Flat_train

# python -u train_gbdt.py -train 1 -thresh 2 -db SPACEV10M -idx IVF1024,Flat

# python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db SPACEV10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=100 > result/result_SPACEV10M_IVF1024_Flat_tree6_b1

# For glove1M
python -u bench_learned_termination.py -mode -1 -batch 10000 -cluster 1024 -thread 10 -thresh 2 -db GLOVE1M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_GLOVE1M_IVF1024_Flat_test

python -u bench_learned_termination.py -mode -2 -batch 100 -train 1 -cluster 1024 -thread 10 -thresh 2 -db GLOVE1M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_GLOVE1M_IVF1024_Flat_train

python -u train_gbdt.py -train 1 -thresh 2 -db GLOVE1M -idx IVF1024,Flat

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db GLOVE1M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=100 > result/result_GLOVE1M_IVF1024_Flat_tree6_b1


# For text10M
python -u bench_learned_termination.py -mode -1 -batch 10000 -cluster 1024 -thread 10 -thresh 2 -db TEXT10M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_TEXT10M_IVF1024_Flat_test

python -u bench_learned_termination.py -mode -2 -batch 100 -train 1 -cluster 1024 -thread 10 -thresh 2 -db TEXT10M -idx IVF1024,Flat -param search_mode=1,pred_max=1024 > result/result_TEXT10M_IVF1024_Flat_train

python -u train_gbdt.py -train 1 -thresh 2 -db TEXT10M -idx IVF1024,Flat

python -u bench_learned_termination.py -mode 1 -batch 1 -cluster 1024 -thresh 2 -db TEXT10M -idx IVF1024,Flat -param search_mode=2,pred_max=1024,nprobe=100 > result/result_TEXT10M_IVF1024_Flat_tree6_b1


# sudo rm /data/zzl/workspace/faiss-learned-termination/benchs/learned_termination/trained_index/
# sudo rm /data/zzl/workspace/faiss-learned-termination/benchs/learned_termination/populated_index/
# sudo rm /data/zzl/workspace/faiss-learned-termination/benchs/learned_termination/ground_truth/

