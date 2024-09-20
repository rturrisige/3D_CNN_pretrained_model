# 3D-CNN pre-trained model for brain T1-weighted MRI

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![pytorch](https://img.shields.io/badge/PyTorch-1.12.1-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

The model has been pre-trained on ADNI dataset, performing AD diagnosis based on brain T1-weighted Magnetic 
Resonance Imaging (MRI).
This project aims at providing a publicly available pre-trained model on 3D MRI 
to perform inference and feature extraction.

- ### INFERENCE

    The pre-trained model can be adopted for AD classification on external datasets. 
An example of how to perform inference can be found in `run_inference_AD_pretrained.sh`.

- ### FEATURE EXTRACTION
   The pre-trained model can be used to extract abstract features and create DL-based MRI embeddings.
An example of how to perform it can be found in `extract_embeddings_AD_pretrained.sh`.


Futher, the model can be used for transfer learning.
The architecture can be imported from `AD_pretrained_utilities.CNN` 
and the weights are provided as `AD_pretrained_weights.pt`. 


More information about the model pre-training can be found in `AD_pretraining_info.txt` and in the full article [Deep learning-based Alzheimer's disease detection: reproducibility and the effect of modeling choices.
](https://doi.org/10.3389/fncom.2024.1360095)

### Requirements 
The codes have been tested on Python 3.8 and PyTorch 1.12.1. 
In order to run, the following Python modules       
are required:

- Numpy, SciPy, Sklearn, seaborn, statistics
- alive_progress
- os, sys, argparse, glob
- PyTorch

# CITATION

Please if you use the pre-trained model, cite it as 

@article{turrisi18deep,

  title={Deep Learning-based Alzheimer's Disease detection: reproducibility and the effect of modeling choices},
  
  author={Turrisi, Rosanna and Verri, Alessandro and Barla, Annalisa},
  
  journal={Frontiers in Computational Neuroscience},
  
  volume={18},
  
  pages={1360095},
  
  publisher={Frontiers}
  
}

