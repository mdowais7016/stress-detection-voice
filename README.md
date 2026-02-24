# Stress Detection from Voice

## Overview
This project detects stress from voice using prosodic and acoustic features.

## Features Used
- Pitch
- Energy (RMS)
- Zero Crossing Rate
- MFCC

## Model
Random Forest Classifier

## Dataset Structure
data/raw/
    stress/
    non_stress/

## Training
python backend/train.py

## Real-Time Prediction
python backend/predict.py
