# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    LabelEncoder, 
    OrdinalEncoder, 
    OneHotEncoder,
    PolynomialFeatures,
    MinMaxScaler,
    StandardScaler,
    RobustScaler
)
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
import catboost as ctb
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.under_sampling import TomekLinks
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load training and test datasets from CSV files
train_data = pd.read_csv('/Users/projects_archieve/loan-prediction/playground-series-s4e10/train.csv')
test_data = pd.read_csv('/Users/projects_archieve/loan-prediction/playground-series-s4e10/test.csv')

# Load additional original dataset
original_data = pd.read_csv('/Users/projects_archieve/loan-prediction/credit_risk_dataset.csv')

# Define categorical columns for later use
cat_cols = [
    'person_home_ownership',
    'loan_intent',
    'cb_person_default_on_file',
    'loan_grade'
] 