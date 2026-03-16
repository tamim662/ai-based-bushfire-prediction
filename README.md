# Multi-Modal Deep Learning and Machine Learning Approaches for Satellite-Based Bushfire Prediction: A Comparative Study Using MODIS Thermal Imagery and ERA5 Weather Data

## Bushfire Prediction Using Satellite and Weather Data

This project investigates early bushfire prediction using satellite thermal imagery and meteorological data combined with machine learning and deep learning models. The system integrates MODIS Land Surface Temperature (LST) imagery and ERA5-Land weather variables to identify environmental conditions that precede wildfire ignition.

The project builds a complete pipeline starting from raw satellite fire detections to model training and evaluation. Historical fire events are extracted from VIIRS active fire detections and converted into a structured dataset through spatio-temporal clustering, positive sample generation, and hard negative sampling.

Multiple modelling approaches are evaluated, including classical machine learning algorithms and deep learning architectures that learn from spatio-temporal satellite imagery and weather sequences.

---

## Project Overview

The project implements an end-to-end workflow for bushfire prediction that includes:

- Extraction of fire events from VIIRS satellite detections
- Spatio-temporal clustering of fire detections into individual events
- Generation of positive samples prior to fire ignition
- Generation of hard negative samples
- Feature extraction from satellite imagery and meteorological time series
- Training and evaluation of machine learning and deep learning models

The objective is to determine whether environmental signals observed several days before ignition can be used to predict bushfire occurrence.

---

## Data Sources

### MODIS Land Surface Temperature (LST)

- MOD11A1 (Terra)
- MYD11A1 (Aqua)
- 1 km spatial resolution
- Daytime and nighttime thermal observations

### ERA5-Land Meteorological Data

Weather variables include:

- Temperature
- Wind components (u10, v10)
- Relative humidity
- Vapor pressure deficit
- Hot-Dry-Windy index

Weather observations are collected over a 7-day window prior to each sample, with data sampled at 3-hour intervals.

### VIIRS Active Fire Data

VIIRS satellite detections are used to identify historical fire events and generate labels for training and evaluation.

---

## Models Implemented

### Classical Machine Learning

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- Support Vector Machine (RBF kernel)

These models use engineered tabular features derived from satellite and weather data.

### Deep Learning Models

- ConvLSTM
- 3D CNN + LSTM
- FireFormer (Transformer-based architecture)

Deep learning models operate directly on spatio-temporal satellite imagery and meteorological sequences.

---

## Evaluation Strategy

### Event-Based Data Splitting

To prevent data leakage, all samples belonging to the same fire event are assigned to the same dataset split.

Dataset split:

- 70% training
- 15% validation
- 15% test

### Cross Validation

Group K-Fold cross-validation with K = 5 is used, where the fire event ID serves as the grouping variable.

### Evaluation Metrics

Model performance is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Specificity
- ROC-AUC

---

## Pipeline Overview

1. Download VIIRS fire detection data  
2. Identify fire events through spatial and temporal clustering  
3. Generate positive samples prior to ignition  
4. Generate negative samples using controlled sampling strategies  
5. Extract MODIS LST and ERA5 weather features  
6. Train machine learning and deep learning models  
7. Evaluate model performance

---

