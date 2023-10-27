# Nanax 
> 🚧: this project is a work in progress
### Nano + JAX

Minimal implementations of various deep learning architectures and training procedures in JAX. Heavily inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl), offering simple and clean implementations with the goal of rapid prototyping for further research.

### Roadmap
- [X] Generative Pre-trained Transformer ([GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf))
- [X] Vision Transformer ([ViT](https://arxiv.org/abs/2010.11929))
- [X] Image Joint Embedding Predictive Architecture ([I-JEPA](https://arxiv.org/abs/2301.08243)) 
- [ ] Contrastive Language Image Pre-training ([CLIP](https://arxiv.org/abs/2103.00020))
- [ ] Vector Quantized Variational Autoencoders ([VQ-VAE](https://arxiv.org/abs/1711.00937))
- [ ] Hierarchical VQ-VAE ([VQ-VAE 2](https://arxiv.org/abs/2002.08111))
- [ ] Denosing Diffusion Probablistic Models ([DDPM](https://arxiv.org/abs/2006.11239))


## Quickstart
Run setup.sh with backend argument to setup virtual environment with required dependencies. 

```
git clone https://github.com/wbrenton/nanax.git
cd nanax
bash setup.sh {cpu, gpu, tpu}
``````