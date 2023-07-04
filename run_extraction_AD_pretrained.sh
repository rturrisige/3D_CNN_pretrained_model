#!/bin/bash

# ARGUMENTS
# -----------------------------
# DATA_DIR: str
# 	it must contain npy files. Each npy file is a list [numpy tensor, int].
# 	The first element is the numpy tensor representing the MRI and the second one (int) is the corresponding label.
# SAVER_DIR: str
# 	it is the path to the folder in which data will be saved.
# EMBEDDING: int between 0 and 7
#	it refers to the convolutional layer to use as feature extractor.
# IMG_PROCESSING: bool
#	input data needs to have size 96 x 96 x 73. 
# 	If IMG_PROCESSING=True, data processing is performed to transform data in a suitable format.

# Path and constants definition
DATA_DIR=./
SAVER_DIR=./
EMBEDDING=6
IMG_PROCESSING=True

echo " "
echo "Extract embedding from AD pretrained model."
echo "Data directory: " $DATA_DIR
echo "Saver directory: " $SAVER_DIR
echo "Embedding to extract=" EMBEDDING " image pre-processing="$IMG_PROCESSING
echo " "

time CUDA_VISIBLE_DEVICES=0 python -u extract_embeddings_AD_pretrained.py \
	--data_dir $DATA_DIR \
	--saver_dir $SAVER_DIR \
	--embedding $EMBEDDING \
	--preprocessing IMG_PROCESSING \
