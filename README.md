# perception4all
This repo aims at implementing and explaining the neural networks in detail
## Table of contents
* [Introduction](#introduction)
* [Technologies](#technologies)
* [Setup](#setup)
* [Contact us](#contact-us)
* [Sources](#sources)

## Introduction

## Technologies
Project is created with:
* Python 3.7.6
* tensorboard 2.1.0  
* torch 1.6.0              
* torchvision 0.5.0 
* matplotlib 3.1.3 
* numpy 1.18.1 


## Setup
To run this project, install it locally:
git clone https://github.com/thariqkhalid/perception4all.git
* install all the required libraries [Technologies](#technologies)
* cd ../perception4all/classification


* To run VGG and resnet networks
* - choose net=VGG.VggNet()or net = resnet.ResNet() in the training file and put the other networks as a comment. 
- finally, run the following command python training.py --exp_name (put a name from your choice).
Example:  python training.py --exp_name expr1

* To run vanilla_cnn network  
- remove transforms.Resize(224) from data_loader.py file.
- choose net=vanilla_cnn.Net() in the training file and put the other networks as a comment. 
- finally, run the following command python training.py --exp_name (put a name from your choice).
Example:  python training.py --exp_name expr1 

* To run inception network 
- run the following command python inception_training.py --exp_name (put a name from your choice).
Example:  python inception_training.py --exp_name expr1

For visualization, write the following command in a new terminal window:
- tensorboard --logdir=experiments/runs

```
* run the following command ...


```
## Contact us
* Afnan Qalas - https://www.linkedin.com/in/afnanbalghaith/
* Thariq Khalid - https://www.linkedin.com/in/thariqkhalid-deeplearning/

## Sources
This project is inspired by https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
- Visualization - https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
- VGG network - https://arxiv.org/pdf/1409.1556.pdf
- Resnet - https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
- Inception network - https://arxiv.org/pdf/1409.4842.pdf

