import pandas as pd
import numpy as np
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from imblearn.under_sampling import TomekLinks
from config import CAT_COLS

class DataProcessor:
    """Class for handling data preprocessing tasks"""
    
    def __init__(self, train_data, test_data, original_data):
        """
        Initialize DataProcessor with raw datasets
        
        Args:
            train_data: Training dataset
            test_data: Test dataset 
            original_data: Original credit risk dataset
        """
        self.train = train_data
        self.test = test_data
        self.original = original_data
        
    def prepare_data(self):
        """Prepare and clean the datasets"""
        # Combine training data
        train = pd.concat([self.train, self.original], axis=0, ignore_index=True)
        
        # Remove ID columns
        train.drop('id', axis=1, inplace=True)
        self.test.drop('id', axis=1, inplace=True)
        
        # Remove outliers
        train.drop(train[train['person_age']>=100].index, axis=0, inplace=True)
        train.drop(train[train['person_emp_length']>=100].index, axis=0, inplace=True)
        
        # Handle missing values
        train = self._impute_missing_values(train)
        
        return train, self.test
    
    def _impute_missing_values(self, df):
        """
        Impute missing values in the dataset
        
        Args:
            df: DataFrame with missing values
            
        Returns:
            DataFrame with imputed values
        """
        # Create numeric loan grade for imputation
        df['loan_grade1'] = df['loan_grade'].replace({
            'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1
        })
        
        # Impute loan interest rate using iterative imputer
        imputer = IterativeImputer(
            estimator=LinearRegression(), 
            max_iter=500, 
            random_state=0
        )
        X_imputed = imputer.fit_transform(
            df[["loan_grade1", "loan_int_rate"]]
        )[:,1]
        df['loan_int_rate1'] = X_imputed
        df['loan_int_rate'].fillna(df['loan_int_rate1'], inplace=True)
        
        # Fill employment length with mean
        df['person_emp_length'] = df['person_emp_length'].fillna(
            df['person_emp_length'].mean()
        )
        
        # Drop temporary columns
        df.drop(['loan_grade1','loan_int_rate1'], axis=1, inplace=True)
        
        return df 