# Deep-Learning-GANs
Python program using PyTorch for implementing Generative Adversarial Networks (GAN) and Deep Convolutional GAN (DCGAN) for MNIST, Cifar10 and STL10 datasets.

## Requirements
- python3
- pytorch and torchvision
```bash
pip install torch torchvision
```

## Highly Recommended
- A CUDA capable GPU

## Usage
The default setting is
- batch size: 200
- number of epochs: 100
- GAN's type: DCGAN
- dataset: MNIST

The command is as follows:
```bash
python3 main.py
```
Also, the arguments can be used as follows:
```bash
python3 main.py -b 100 -e 20 -gt gan -d cifar10
```
- -b or --batch: batch size
- -e or --epochs: number of epochs
- -gt or --gantype: gan type (gan or dcgan)
- -d or --dataset: dataset (mnist, cifar10 and stl10)
- -ngf: number of Generator features
- -ndf: number of Discriminator features

## Note:
Issue currently experiencing issue with DCGAN on MNIST10 dataset due to grayscale images.

## References
- [Ian J. Goodfellow, et al, Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Ian J. Goodfellow, et al, Improving techniques for GANs](https://arxiv.org/abs/1606.03498)
- [Alec Radford, et al, Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [Phillip Isola, et al, Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
