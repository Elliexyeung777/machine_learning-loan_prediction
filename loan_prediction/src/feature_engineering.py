def feature_eng(train, test):
    """
    Create new features from existing ones
    
    Args:
        train: Training dataset
        test: Test dataset
        
    Returns:
        Tuple of transformed training and test datasets with new features
    """
    df = train.copy()
    df_test = test_data.copy()
    
    # Calculate loan income error (difference between calculated and stated loan percent income)
    df['loan_income_err'] = (df['loan_amnt']/df['person_income']).round(3) - df['loan_percent_income']
    df_test['loan_income_err'] = (df_test['loan_amnt']/df_test['person_income']).round(3) - df_test['loan_percent_income']
    
    return df, df_test

def cat_boost_encoding(train):
    """
    Prepare features for CatBoost model by converting to appropriate types
    
    Args:
        train: Input dataset
        
    Returns:
        DataFrame with all features converted to categorical type
    """
    train_enc = train.copy()
    
    # Convert numeric columns to string type for categorical encoding
    numeric_conversions = {
        'person_age': 'int',
        'person_emp_length': 'int',
        'person_income': 'int',
        'loan_int_rate': 'float',
        'loan_percent_income': 'float',
        'cb_person_cred_hist_length': 'int'
    }
    
    for col, dtype in numeric_conversions.items():
        train_enc[col] = train_enc[col].astype(dtype).astype('string')
    
    # Handle special case for loan_income_err if it exists
    if 'loan_income_err' in train_enc.columns:
        train_enc['loan_income_err'] = (train_enc['loan_income_err']*1000).astype(int).astype('string')
    
    # Convert all columns to category type
    for col in train_enc.columns:
        train_enc[col] = train_enc[col].astype('category')
        
    return train_enc 