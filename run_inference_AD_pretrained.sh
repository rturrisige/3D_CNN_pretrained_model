#!/bin/bash

# ARGUMENTS
# -----------------------------
# DATA_DIR: str
# 	must contain npy files. Each npy file contains the numpy tensor representing the MRI.
# SAVER_DIR: str
# 	it is the path to the folder in which data will be saved.
# BATCH_SIZE: int
#	data batch size 
# IMG_PROCESSING: bool
#	input data needs to have size 96 x 96 x 73. 
# 	If IMG_PROCESSING=True, data processing is performed to transform data in a suitable format.

# Path and constants definition
DATA_DIR=./
SAVER_DIR=./
IMG_PROCESSING=False
BATCH_SIZE=50

echo " "
echo "Inference from AD pretrained model."
echo "Data directory: " $DATA_DIR
echo "Saver directory: " $SAVER_DIR
echo "Batch size=" $BATCH_SIZE " image pre-processing="$IMG_PROCESSING
echo " "

time CUDA_VISIBLE_DEVICES=0 python -u inference_AD_pretrained.py \
	--data_dir $DATA_DIR \
	--saver_dir $SAVER_DIR \
	--batch_size $BATCH_SIZE \
	--preprocessing IMG_PROCESSING \
