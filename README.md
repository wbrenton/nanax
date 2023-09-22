# Nanax 
> **Warning**: this project is a work in progress
### Minimal implementations in JAX (nano + JAX)


Minimal implementations of various deep learning architectures and training procedures in JAX (Flax). Heavily inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl), with simple and clean implementations for buiding research on top of.

### Roadmap
- [X] Generative Pre-trained Transformer ([GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [X] Vision Transformer ([ViT](https://arxiv.org/abs/2010.11929)
- [X] Image Joint Embedding Predictive Architecture ([i-JEPA](https://arxiv.org/abs/2301.08243)) 
- [ ] Contrastive Language Image Pre-training ([CLIP](https://arxiv.org/abs/2103.00020))
- [ ] Vector Quantized Variational Autoencoders ([VQ-VAE](https://arxiv.org/abs/1711.00937))
- [ ] Hierarchical VQ-VAE ([VQ-VAE 2](https://arxiv.org/abs/2002.08111))
- [ ] Denosing Diffusion Probablistic Models ([DDPM](https://arxiv.org/abs/2006.11239))


## Quickstart
Nanax comes with cpu.sh, gpu.sh, tpu.sh, so you can quickly get setup 
```
git clone https://github.com/wbrenton/nanax.git
cd nanax
bash cpu.sh # {gpu.sh, tpu.sh}
```

