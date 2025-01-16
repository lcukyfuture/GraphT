# Codes Information and Running step

## Installation
```
python=3.9
numpy
scipy
torch=2.2.0
torch-geometric=2.6.1
torch-cluster=1.6.3
torch-scatter=2.1.2
torch-sparse=0.6.18
einops
matplotlib

```


## Run experiments
```bash

python classification.py --dataset MUTAG --kernels WL RW --hop 2 --wl 3 --outdir MUTAG

```
