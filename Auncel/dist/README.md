## 1. Dataset<br>

This distributed experiment is based on [DEEP1B](https://research.yandex.com/datasets/biganns) dataset which 
consists of 109 image embeddings produced as the outputs from the last fully-connected layer of the GoogLeNet model.

Since the experiment adopts different number of workers, you should shard the 1B dataset according to the corresponding
number of workers (run `split.cpp`). After sharding, you shoule generate the ground truth for each shard on the trainset 
queries (run `gt.cpp`).

## 2. Environment requirement<br>

- Hardware<br>
    - Four AWS c5.metal machines with 96 vCPU and 192 GiB host memory each to support the larger number of workers and hold the large dataset.
- Software<br>
    - Intel MKL<br> & CMAKE >= 1.17 & OpenMP
- Settings<br>
    - To connect the differet machines, you should change the network settings in AWS website to support TCP protocol.

## 3. How to run<br>

- Master<br>
    Choose a machine as master node, and run `master.cpp`. It will automaticly spawns a process as master worker.
- Worker<br>
    Run `worker.cpp` on the each machine for $Num/4$  times. The workers communicate with the the master through TCP
    and sycn with each other.