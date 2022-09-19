# PACFL

The official code of paper [PACFL].

In this repository, we release the official implementation for PACFL algorithm. We also release the implementation of the following algorithms:
* FedAvg
* FedProx
* FedNova
* Scaffold
* Per-FedAvg
* IFCA
* LG-FedAvg
* CFL
* MTL 
* pFedMe
* SOLO


## Usage

We provide scripts to run the algorithms, which are put under `scripts/`. Here is an example to run the script:
```
cd scripts
bash pacfl.sh
```
Please follow the paper to modify the scripts for more experiments. You may change the parameters listed in the following table.

The descriptions of parameters are as follows:
| Parameter | Description |
| --------- | ----------- |
| ntrials      | The number of total runs. |
| rounds       | The number of communication rounds per run. |
| num_users    | The number of clients. |
| frac         | The sampling rate of clients for each round. |
| local_ep     | The number of local training epochs. |
| local_bs     | Local batch size. |
| lr           | The learning rate for local models. |
| momentum     | The momentum for the optimizer. |
| model        | Network architecture. Options: `TODO` |
| dataset      | The dataset for training and testing. Options are discussed above. |
| partition    | How datasets are partitioned. Options: `homo`, `noniid-labeldir`, `noniid-#label1` (or 2, 3, ..., which means the fixed number of labels each party owns). |
| datadir      | The path of datasets. |
| logdir       | The path to store logs. |
| log_filename | The folder name for multiple runs. E.g., with `ntrials=3` and `log_filename=$trial`, the logs of 3 runs will be located in 3 folders named `1`, `2`, and `3`. |
| alg          | Federated learning algorithm. Options are discussed above. |
| beta         | The concentration parameter of the Dirichlet distribution for heterogeneous partition. |
| local_view   | If true puts local test set for each client |
| gpu          | The IDs of GPU to use. E.g., `TODO` |
| print_freq   | The frequency to print training logs. E.g., with `print_freq=10`, training logs are displayed every 10 communication rounds. |

## MIX-4 
We have also released the codes for MIX-4 experiments in the paper under mix4 folder. Please follow the same instruction as in usage to run the scripts for each algorithm. 

## Generalization to Unseen Clients
We have also released the codes for the generalization to unseen clients experiments in the paper under unseen_clients folder. Please follow the same instruction as in usage to run the scripts for each algorithm. 

## Citation 
Please cite our work if you find it relavent to your research and used our implementations. 
```
. 
```

## Acknowledgements

Some parts of our code and implementation has been adapted from [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench) repository.

## Contact 
If you had any questions, please feel free to contact me at mmorafah@eng.ucsd.edu
