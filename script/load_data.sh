#!/bin/bash

source ~/pythonenv/kagglebase/bin/activate
kaggle competitions download -c asl-fingerspelling
unzip asl-fingerspelling.zip -d data/asl-fingerspelling
rm asl-fingerspelling.zip