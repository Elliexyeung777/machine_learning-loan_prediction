# Combine training data with original data and remove ID column
train = pd.concat([train_data, original_data], axis=0, ignore_index=True)
train.drop('id', axis=1, inplace=True)
test_data.drop('id', axis=1, inplace=True)

# Remove outliers for age and employment length
train.drop(train[train['person_age']>=100].index, axis=0, inplace=True)
train.drop(train[train['person_emp_length']>=100].index, axis=0, inplace=True)

# Handle missing values
# Convert loan grades to numeric values for imputation
train['loan_grade1'] = train['loan_grade'].replace({
    'A':7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1
})

# Use iterative imputation for loan interest rate
imputer = IterativeImputer(estimator=LinearRegression(), max_iter=500, random_state=0)
X_imputed = imputer.fit_transform(train[["loan_grade1", "loan_int_rate"]])[:,1]
train['loan_int_rate1'] = X_imputed
train['loan_int_rate'].fillna(train['loan_int_rate1'], inplace=True)

# Fill missing employment length with mean
train['person_emp_length'] = train['person_emp_length'].fillna(train['person_emp_length'].mean())

# Remove temporary columns used for imputation
train.drop(['loan_grade1','loan_int_rate1'], axis=1, inplace=True)