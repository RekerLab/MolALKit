# ActiveLearningBenchmark
Benchmark for molecular active learning.

## Installation
```commandline
conda env create -f environment.yml
conda activate alb
```
[GPU-enabled PyTorch](https://pytorch.org/get-started/locally/).
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
## Yoked Learning
reproduce all results of [Yoked Learning in Molecular Data Science]() using following command:
```commandline
bash YoL.sh
```
