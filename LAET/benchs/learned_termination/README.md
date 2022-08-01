# Benchmarking Faiss with learned adaptive early termination

This directory contains benchmarking scripts and necessary files to reproduce the experiments described in our SIGMOD paper.

## Get datasets

You need to download the ANN_SIFT1B and ANN_GIST1M datasets from http://corpus-texmex.irisa.fr/.

The DEEP1B dataset is available at https://yadi.sk/d/11eDCm7Dsn9GA. For the learning and database vectors, use the script https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py to download the data to subdirectory deep1b/, then concatenate the database files to base.fvecs and the training files to learn.fvecs.

The SIFT1B and DEEP1B datasets are huge, so you might want to start from GIST1M.

We didn't use the public ground truth files because our accuracy metric allows tied nearest neighbors. We provided our own ground truth files, and [compute_gt.py](compute_gt.py) is used to find the ground truth nearest neighbors.

## Reproduce experiments

[`run.sh`](run.sh), [`bench_learned_termination.py`](bench_learned_termination.py), and [`train_gbdt.py`](train_gbdt.py) are the main scripts we used to run the experiments. We also provided LightGBM models that we trained so that you can rerun our experiments without training your own models.

## One issue about HNSW index

In order to reproduce the experiments, one requirement is that you need to build the exactly same ANN index as we used. For IVF index (with or without quantization), we provided the *trained.index files (which include the trained centroids and quantization codebook) so that you are able to build the exact same index as us. However, for HNSW index the only way to build the exact same index is to use the same *populated.index files. However, those files can be as large as tens of GBs so we cannot upload them to GitHub. Instead, we uploaded the *populated.index file for the GIST1M dataset at here in case you want to strictly reproduce our experiment: 

https://drive.google.com/file/d/13q4kNCazU25pfVLVqpWR5kNr10S0f9-J/view?usp=sharing

If you use your own HNSW *populated.index file, then the LightGBM models we provided might not be able to provide the same level of latency reduction since they are trained based on a potentially different indexing. In this case you need to train you own prediction model.
