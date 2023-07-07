## 1. Introduction<br>
This repository contains one version of the source code for our NSDI'23 paper "Fast, Approximate Vector Queries on Very Large Unstructured Datasets" [[Paper]](https://www.usenix.org/conference/nsdi23/presentation/zhang-zili-0).


## 2. Content<br>

- Auncel/<br>
    - The source code of Auncel implementation and design (fork from [Faiss 1.15.2](https://github.com/facebookresearch/faiss/tree/v1.5.2))
- LAET/<br>
    - The source code of sigmod20 paper, "Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination" (fork from [LAET](https://github.com/efficient/faiss-learned-termination) and add new datasets)
- faiss/<br>
    - The source code of vector search engine, Faiss (fork from [Faiss 1.15.2](https://github.com/facebookresearch/faiss/tree/v1.5.2) and change its ELP (`Autotune.cpp`) from average case to bounded case)

## 3. Environment requirement<br>

- Hardware<br>
  - AWS c5.4xlarge & c5.metal<br>
- Software<br>
  - Intel MKL & clang & OpenMP<br>
- Datasets<br>  
    - The 10M-dataset is a random 10M slice of the whole 1B-dataset ([SIFT](http://corpus-texmex.irisa.fr/) [DEEP](https://research.yandex.com/datasets/biganns) [TEXT](https://big-ann-benchmarks.com/) [GIST](http://corpus-texmex.irisa.fr/)). You can download the preprocessed(e.g., normalized for text) datasets here [data-link-1](https://disk.pku.edu.cn/#/link/A872A42BA875127DB9DC940A6557B1E6) or [data-link-2](https://pan.baidu.com/s/13HuAqeyTXWduBopm22187g)(7w3r) (I recommend you to use the provided datasets if you want to use our configuration)

## 4. How to run<br>

- Compile<br>
    - Run the following commands: `cd ./Auncel && ./configure --without-cuda && ./build.sh && cd ../` to compile the code of Auncel
    - Run the following commands: `cd ./LAET && ./configure --without-cuda && ./build.sh && cd ../` to compile the code of LAET
    - Run the following commands: `cd ./faiss && ./configure --without-cuda && ./build.sh && cd ../` to compile the code of Faiss
- Run
    - **Overall** : Before running the python programs to generate the figures, you are supposed to run the corresponding program to get result log files. Run `cd ./Auncel/eval/ && ./run.sh && cd -` to get log files of Auncel.
    Run `cd ./LAET/benchs/learned_termination/ && ./run.sh && cd -` to get log files of LAET. Run `cd ./faiss/eval/run.sh && && ./run.sh && cd -` to get log files of Faiss. 
    Run `cd ./figures/overall/ && ./overall.sh && cd -` to get the three figures.
    - **Effectiveness** : Before running the python programs to generate the figures, you are supposed to run the corresponding program to get result log files. Run `cd ./Auncel/eval/ && ./effect.sh && cd -` to get log files of Auncel.
    Run `cd ./figures/effect/ && ./effect.sh && cd -` to get the two figures.
    - **Validation** : The log files are automatically generated when you run `cd ./Auncel/eval/ && ./run.sh && cd -`. 
    (Please set `<repo>/Auncel/IVF_pro.h/struct Trace -> bs` as 1 to capture every point in the $\varphi - U$ map. ) 
    To draw the figures, please run `cd ./figures/validation && ./validation.sh && cd -`.
    - **Overhead** : Run `cd ./Auncel/eval/ && ./overhead.sh && cd -` and you will get the corresponding experimental data on the terminal.
    - **Dist** : Please refer `<repo>/Auncel/dist/README.md` for the details of distributed experiment. The figure script is `<repo>/figures/dist/figure16.py`

## 5. Contact<br>

For any question, please contact  `zzlcs at pku dot edu dot cn`.
