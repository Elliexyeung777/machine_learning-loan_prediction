# Loan Default Prediction Project

## Overview
This project implements a machine learning solution to predict loan default risk. The model analyzes borrower information, financial status, and loan characteristics to assess the probability of loan default.

## Project Structure
loan-prediction/
├── src/
│ ├── config.py # Configuration settings
│ ├── data_loading.py # Data loading and library imports
│ ├── data_preprocessing.py # Data preprocessing functions
│ ├── data_processor.py # Data processing class
│ ├── feature_engineering.py # Feature engineering
│ ├── main.py # Main execution script
│ ├── model_params.py # Model parameters
│ ├── model_training.py # Training implementation
│ ├── model.py # Model definition
│ └── prediction.py # Prediction functionality
│
├── loan_prediction.ipynb # Jupyter notebook with development process
└── README.md


## Technical Implementation

### Data Pipeline
- **config.py**: Contains configuration settings and parameters
- **data_loading.py**: Handles data import and initial setup
- **data_preprocessing.py**: Implements data cleaning and preprocessing
- **data_processor.py**: Main class for data processing operations
- **feature_engineering.py**: Creates and transforms features

### Model Pipeline
- **model.py**: Defines model architecture
- **model_params.py**: Stores model hyperparameters
- **model_training.py**: Implements training and validation
- **prediction.py**: Handles prediction generation

## Requirements
- Python 3.11+
- Key Dependencies:
  - numpy
  - pandas
  - scikit-learn
  - catboost
  - imbalanced-learn
  - matplotlib
  - seaborn
