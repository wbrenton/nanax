# Nanax 
> **Warning**: this project is a work in progress
### Minimal implementations in JAX (nano + JAX)


Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [cleanRL](https://github.com/vwxyzjn/cleanrl), Nanax aims to create easy to understand toy implementations of various algorithms/architectures.

### 1. Self Contained
  - each implementation contains train.py, model.py, and a data folder
  - freeing you from having to familiarize yourself with the entire project
### 2. Uniform Format
  - in addition to uniform file structure, the design of each script across implementations reuses much of the code
  - enabling faster understanding of other implementations once familiar with one
### 3. Accessibility
  - "What I cannot create, I do not understand." 
  - implementations have minimal compute requirements, allowing you train from scratch on your own or easily fork and train on your own data

### Roadmap
- [X] Generative Pre-trained Transformer (GPT)
- [ ] Vision Transformer (ViT)
- [ ] Image Joint Embedding Predictive Architecture
- [ ] Contrastive Language Image Pre-training


## Quickstart
Nanax comes with cpu.sh, gpu.sh, tpu.sh, so you can quickly get setup 
```
git clone https://github.com/wbrenton/nanax.git
cd nanax
bash cpu.sh # {gpu.sh, tpu.sh}
```
