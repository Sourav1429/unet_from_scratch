# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:05:32 2023

@author: Sourav
"""
import os
import torch

#Defining the base path
base_path = "D://datasets//Unet_seg//UNET_from_scratch"
DATASET_PATH = os.path.join(base_path,"dataset","train")
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH,"original")
MASK_DATASET_PATH = os.path.join(DATASET_PATH,"mask")

TEST_SPLIT = 0.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if DEVICE=='cuda' else False

NUM_CHANNELS,NUM_CLASSES,NUM_LEVELS = 3,1,3

INIT_LR,NUM_EPOCHS,BATCH_SIZE = 0.001,5,1

INPUT_IMAGE_HEIGHT,INPUT_IMAGE_WIDTH = 128,128
THRESHOLD = 0.5

#o/p directories
BASE_OUTPUT = "output"
MODEL_PATH = os.path.join(BASE_OUTPUT,"best.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT,'plot.png'])
TEST_PATH = os.path.sep.join([BASE_OUTPUT,'test_paths.txt'])
