import pandas as pd
from catboost import CatBoostClassifier
from config import *
from data_processor import DataProcessor
from feature_engineering import FeatureEngineer
from model import CVTrainer

def main():
    # Load data
    train_data = pd.read_csv('/Users/projects_archieve/loan-prediction/playground-series-s4e10/train.csv')
    test_data = pd.read_csv('/Users/projects_archieve/loan-prediction/playground-series-s4e10/test.csv')
    original_data = pd.read_csv('/Users/projects_archieve/loan-prediction/playground-series-s4e10/application_train.csv')
    
    # Process data
    processor = DataProcessor(train_data, test_data, original_data)
    train, test = processor.prepare_data()
    
    # Extract target variable
    y = train['loan_status']
    train.drop('loan_status', axis=1, inplace=True)
    
    # Feature engineering
    train_fe, test_fe = FeatureEngineer.create_features(train, test)
    train_cat = FeatureEngineer.prepare_catboost_features(train_fe)
    test_cat = FeatureEngineer.prepare_catboost_features(test_fe)
    
    # Initialize model
    model_params = CATBOOST_PARAMS.copy()
    model_params['cat_features'] = train_cat.columns.to_list()
    model_params['random_state'] = 42
    model = CatBoostClassifier(**model_params)
    
    # Train model
    trainer = CVTrainer(
        model=model,
        train_data=train_cat,
        target_data=y,
        test_data=test_cat,
        n_split=10
    )
    _, test_pred, _ = trainer.train_model()
    
    # Create submission
    submission = pd.read_csv(SUBMISSION_PATH)
    submission['loan_status'] = test_pred
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main() 