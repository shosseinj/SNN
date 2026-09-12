# TTFS Spiking Neural Network Experiments

This repository is a personal experimental working copy based on the code released with **“High-performance deep spiking neural networks with 0.3 spikes per neuron”** by Stanojevic et al., *Nature Communications* (2024).

It has been used to study and modify time-to-first-spike (TTFS) neural-network training, ANN-to-SNN mapping, model evaluation, and implementation details around spiking computation.

## Upstream Work

The underlying method and original code are the work of Stanojevic, Woźniak, Bellec, Cherubini, Pantazi, and Gerstner. Please cite the original publication when using the method or baseline implementation.

This repository should not be interpreted as an original implementation of their method.

## Experimental Contents

The working tree includes dataset utilities, model definitions, training variants, evaluation code, batch-normalization fusion experiments, and intermediate implementation branches used during local experimentation.

## Environment

The original codebase uses a Conda environment defined in `environment.yml`. Individual experimental scripts may have additional assumptions documented in their source or in `Steps.md`.

## Research Use

This repository is retained as a baseline/experimental workspace. My more recent spiking ConvNeXt work is separated into dedicated repositories to make the research contributions and experimental provenance clearer.


## Goal

The repository is retained to reproduce, inspect, and modify the published TTFS SNN baseline while keeping that upstream work distinct from the later ConvNeXt-based experiments on this profile.

## Installation

Create the Conda environment supplied with the upstream code and activate the environment name declared in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate tf24
python main.py --help
```

GPU use depends on the TensorFlow and CUDA versions selected by that environment.

## Working with the Repository

`main.py` trains or evaluates the ReLU and SNN variants for MNIST, CIFAR-10, or CIFAR-100. Use explicit values for `--model_type`, `--model_name`, `--data_name`, `--logging_dir`, and `--weight_dir`. Keep upstream baseline results separate from locally produced runs, and cite the original publication when reusing the method or code.
