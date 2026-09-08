# MRI Harmonization using Conditional Diffusion Models

**author** : Benjamin Olive

This repository contains the code developed during a research internship on MRI harmonization using conditional diffusion models.

**Internship context**: CHU Lille - LIIFE research team, March-August 2026, supervised by Renaud LOPES and Barnabé Hache.


## Overview

In MRI studies, pooling data from multiple sites increases sample size. However, this process introduces variability due to differences between acquisition sites and the parameters employed. 

In this context, deep learning methods for image style transfer offer a promising approach to harmonizing MRI images from different sites. This work builds upon the LIIFE team’s research into leveraging the power of diffusion models—guided here by an anatomical representation and a site embedding to reconstruct scans that preserve subject-specific characteristics while modifying the style associated with the acquisition site. 

The aim of this study is to assess whether integrating modules from the MR-CLIP and DIST-CLIP foundation models improves harmonization in a diffusion model by exploiting the large datasets captured by those models. At inference, the model can be applied to any image, even from an unknown acquisition site, making it a universal harmonization generator. 

The impact of replacing the anatomical encoder was evaluated in isolation. While this new representation allows for better disentanglement of style from anatomy, it does not translate into improved harmonization capabilities. This result suggests that merely separating style and anatomy is insufficient to enhance anatomical guidance. 

Experiments regarding style guidance, as well as the joint evaluation of both modules, remain to be conducted to determine the overall contribution of the proposed modifications

### What this project contains

The proposed pipeline combines:
- Using 2D MRI image data; (Architecture could be modified for 3D datasets),
- anatomical representations extracted using a pre-trained encoder,
- domain conditioning,
- a 2D conditional diffusion model for scans harmonization.

### Main objectives

1. Apply diffusion models in MRI harmonization,
2. Preserve anatomical information using model guidance,
3. Compare different anatomical guidance strategies,
4. Design a 2D architecture that is easily convertible to 3D.


## Method

### Anatomical conditioning

Two anatomical representations are available :
- Canny edge map,
- Anatomical representation extracted using [DIST-CLIP](https://github.com/myigitavci/MaRaI).

### Diffusion model

The core of this project is a conditional diffusion model, trained to predict the noise introduced during the diffusion process.

This implementation supports domain conditioning and requires both an anatomical representation and the noised MRI scan as input.

### Workflow

1. Raw MRI volumes,
2. Data preprocessing 
    - 2D slice extraction,
    - Anatomical encoding,
    - [MR-CLIP](https://github.com/myigitavci/MaRaI) input preparation.
3. Conditional diffusion model
    - Inputs :
        - MRI data,
        - Anatomical representation,
        - Site embedding produced by MR-CLIP,
4. Harmonized MRI scans


Harmonised MRI


## Project usage

### Data

The datasets used during this project are **not** distributed in this repository.


### Data Preprocessing
Before running the model, the data must undergo a preprocessing step.
Some of these steps involves the use of pre-trained models such as MR-CLIP or the DIST-CLIP beta encoder.

Full details of this step are available here:
[see data preprocessing](precompute/precompute_instructions.md)


### Configuration

Experiments are controlled using JSON configuration files located in:
_configs/_

These files are used to define parameters of the model (called in the main file):
- dataset paths,
- training parameters,
- diffusion parameters,
- conditioning representation used,
- checkpoint directories,
- settings used in evaluation.

Please make sure that all paths correspond to your environment.

### Training

To launch either training, evaluation, or inference, please modify and execute: [main_diffusion_2d.py](./main_diffusion_2d.py)

Indicate the configuration to use and the training mode to be used in this script, then run:
_accelerate launch main_diffusion_2d.py_

Multi-GPU training can also be used. Checkpoints will be saved during training.