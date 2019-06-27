# Wnet-CGAN For Building Outline

## Introduction

Building outlines  are a relevant information for multiple tasks such as urban planning, disaster assessment 
and cartographic analysis. DSM (Digital Surface Model) depth information is useful for filtering the elevated objects (buildings,...) 
from the objects close to the ground (streets, ...). [(Bittner et al., 2019)](https://arxiv.org/pdf/1903.03519.pdf) Combined very high resolution PAN imagery
with DSM model to refine the 3D building footprints. This Repository presents an implementation of their model using the Keras Framework. <br>
<br>
![Results obtained by [(Bittner et al., 2019)](https://arxiv.org/pdf/1903.03519.pdf)](https://github.com/0xzayd/Wnet-cGAN/blob/master/img/results.png)

## Architecture

The architecture is a conditional Generative Adversarial Neural Network which consists of a generator network which 
consists of the so called W-net that admits two inputs: A panchromatic image and A stereo DSM, stacked with a discriminator that discriminates Real and Fake labels. <br>

![Wnet-cGAN Architecture using depth and spectral information](https://github.com/0xzayd/Wnet-cGAN/blob/master/img/Wnet_cgan.png)

### Generator Network

The Generator network contains two [U-net](https://arxiv.org/pdf/1505.04597.pdf) networks that predict the building footprints. The output of each U-net is fused at the output.


### Discriminator

The Discriminator is a neural network with 5 Convolutional layers that takes the fused output of the generator (or ground truth) and the DSM data to discriminate the fake from real labels. 

### Implementation

In this implementation I train the generator and the discriminator sequentially. I shuffle the pairs (input DSM, Generated label) with the pairs (input DSM, Ground truth Label) and train the discriminator on that dataset. This has shown speed up and stability in training the discriminator.<br>
The hyperparamters can be changed as the original paper gives little information about the hyperparameters.


### Training

training data (tiles of same size, recommended 256x256, single band tif files) must be put in 3 different folders: PAN, DSM and LABEL as shown in the schema below.

```
Training Folder
               ├───DSM
               │   └─── *.tif files
               ├───PAN
               │   └─── *.tif files
               └───LABEL
			       └─── *.tif files
```

to run the training, training folder must be passed as an argument to the train.py script:
```
python train.py --data ./training_folder
```

### Inference

in order to test the pretrained model on a whole tif image, paths to the DSM, PAN input files as well as the weights file and the output file must be passed as arguments to the script predict.p:

```
python predict.py --dsm ./dsm.tif --pan ./pan.tif --weights ./model/weights_final.hdf5 --output ./output_predicted.tif
```



