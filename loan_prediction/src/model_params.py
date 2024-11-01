# CatBoost model parameters
cat_params = {
    'iterations': 3000,          # Number of trees
    'depth': 7,                  # Depth of trees
    'eta': 0.3,                 # Learning rate
    'reg_lambda': 40.0,         # L2 regularization
    'loss_function': 'Logloss', # Loss function for binary classification
    'eval_metric': 'AUC',       # Evaluation metric
    'min_data_in_leaf': 51,     # Minimum samples in leaf nodes
    'early_stopping_rounds': 300,# Early stopping patience
    'cat_features': train_cat.columns.to_list(), # Categorical features
    'task_type': 'CPU',         # Computation device
    'verbose': 200,             # Print period
    'scale_pos_weight': 2.5,    # Weight for positive class
    'use_best_model': True,     # Use best iteration
} 