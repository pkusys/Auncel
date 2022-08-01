## 1. Introduction<br>



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
  - Intel MKL & CMAKE >= 1.17 & OpenMP<br>
- Datasets<br>
  - [SIFT10M](http://corpus-texmex.irisa.fr/)
  - [GIST10M](http://corpus-texmex.irisa.fr/)
  - [DEEP10M](https://research.yandex.com/datasets/biganns)
  - [DEEP1B](https://research.yandex.com/datasets/biganns)
  - [TEXT10M](https://big-ann-benchmarks.com/)

## 4. How to run<br>

- Compile<br>
    - Run the following commands: `cd ./Auncel && ./configure --without-cuda && ./build.sh && cd ../` to compile the code of Auncel
    - Run the following commands: `cd ./LAET && ./configure --without-cuda && ./build.sh && cd ../` to compile the code of LAET
    - Run the following commands: `cd ./faiss && ./configure --without-cuda && ./build.sh && cd ../` to compile the code of Faiss
- Reproduce
    - **Figure 10 - 12** : Before running the python programs to generate the figures, you are supposed to run the corresponding program to get result log files. Run `cd ./Auncel/eval/ && ./run.sh && cd -` to get log files of Auncel.
    Run `cd ./LAET/benchs/learned_termination/ && ./run.sh && cd -` to get log files of LAET. Run `cd ./faiss/eval/run.sh && && ./run.sh && cd -` to get log files of Faiss. 
    Run `cd ./figures/overall/ && ./overall.sh && cd -` to get the three figures.
    - **Figure 13 - 14** : Before running the python programs to generate the figures, you are supposed to run the corresponding program to get result log files. Run `cd ./Auncel/eval/ && ./effect.sh && cd -` to get log files of Auncel.
    Run `cd ./figures/effect/ && ./effect.sh && cd -` to get the two figures.
    - **Figure 15** : The log files are automatically generated when you run `cd ./Auncel/eval/ && ./run.sh && cd -`. 
    To draw the figures, please run `cd ./figures/validation && ./validation.sh && cd -`.
    - **Table 3 - 5** : Run `cd ./Auncel/eval/ && ./overhead.sh && cd -` and you will get the corresponding experimental data on the terminal.
    - **Figure 16** : Please refer `<repo>/Auncel/dist/README.md` for the details of distributed experiment. The figure script is `<repo>/figures/dist/figure16.py`

## 5. Contact<br>

For any question, please contact  `zzlcs at pku dot edu dot cn`.