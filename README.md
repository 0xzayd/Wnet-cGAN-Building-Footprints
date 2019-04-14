# Wnet-CGAN For Building Outline

## Introduction

Building outlines  are a relevant information for multiple tasks such as urban planning, disaster assessment 
and cartographic analysis. DSM (Digital Surface Model) is useful for filtering the elevated objects (buildings,...) 
from the objects close to the ground (streets, ...). [(Bittner et al., 2019)](https://arxiv.org/pdf/1903.03519.pdf) Combined very high resolution PAN imagery
with DSM model to refine the 3D building footprints. This Repository presents an implementation of their model using the Keras Framework.

## Architecture

The architecture is a conditional Generative Adversarial Neural Network which consists of a generator network which 
consists of the so called W-net that admits two inputs: A panchromatic image and A stereo DSM, stacked with a discriminator that discriminates Real and Fake labels.

### Generator Network

The Generator network contains two [U-net](https://arxiv.org/pdf/1505.04597.pdf) networks that predict the building footprints. The output of each U-net is fused at the output.


### Discriminator

The Discriminator is a neural network with 5 Convolutional layers that takes the fused output of the generator (or ground truth) and the DSM data to discriminate the fake from real labels. 

### Implementation

In this implementation I train the generator and the discriminator sequentially. I shuffle the pairs (input DSM, Generated label) with the pairs (input DSM, Ground truth Label) and train the discriminator on that dataset. This has shown speed up and stability in training the discriminator.



