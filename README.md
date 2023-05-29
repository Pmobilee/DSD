# DSD - Direct Self-Distillation

## About

This repository contains a novel approach to distillation of Diffusion Models, as detailed in the (currently unpublished) thesis/paper titled "A Self-help guide for Diffusion Models". This method tries to improve upon previous distillation approaches by distilling a model into itself, without the need for any separately initialised teacher model.

## Conditional Diffusion Model: ImageNet (256x256)

![Cin256](https://github.com/Pmobilee/DSD/blob/main/readme/Cin256.png?raw=true)

## Unconditional Diffusion Model: CelebA-HQ (256x256)

![Celeb](https://github.com/Pmobilee/DSD/blob/main/readme/Celeb.png?raw=true)

## Allocation of weight updates per denoising step-size

![Updates](https://github.com/Pmobilee/DSD/blob/main/readme/Updates.png?raw=true)
