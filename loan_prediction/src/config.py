# Configuration settings for the loan prediction project
import os

# Data paths
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
ORIGINAL_DATA_PATH = 'data/credit_risk_dataset.csv'
SUBMISSION_PATH = 'data/sample_submission.csv'

# Categorical columns
CAT_COLS = [
    'person_home_ownership',
    'loan_intent', 
    'cb_person_default_on_file',
    'loan_grade'
]

# Model parameters for CatBoost
CATBOOST_PARAMS = {
    'iterations': 3000,
    'depth': 7,
    'eta': 0.3,
    'reg_lambda': 40.0,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'min_data_in_leaf': 51,
    'early_stopping_rounds': 300,
    'task_type': 'CPU',
    'verbose': 200,
    'scale_pos_weight': 2.5,
    'use_best_model': True,
} 