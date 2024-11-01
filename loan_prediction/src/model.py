from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
import numpy as np
from imblearn.under_sampling import TomekLinks

class CVTrainer:
    """Class for cross-validation training"""
    
    def __init__(self, model, train_data, target_data, test_data, n_split):
        """
        Initialize CVTrainer
        
        Args:
            model: ML model instance
            train_data: Training features
            target_data: Target variable
            test_data: Test features
            n_split: Number of CV folds
        """
        self.model = model
        self.train = train_data
        self.target = target_data
        self.test = test_data
        self.n_split = n_split
        
    def train_model(self):
        """
        Train model using cross-validation
        
        Returns:
            Tuple of (oof_predictions, test_predictions, cv_score)
        """
        fold = StratifiedKFold(
            n_splits=self.n_split,
            shuffle=True,
            random_state=42
        )
        
        oof_pred = np.zeros(len(self.train))
        test_pred = np.zeros(len(self.test))
        scores = []
        
        for idx_train, idx_val in fold.split(self.train, self.target):
            # Split data
            X_train = self.train.iloc[idx_train]
            X_val = self.train.iloc[idx_val]
            y_train = self.target.iloc[idx_train]
            y_val = self.target.iloc[idx_val]
            
            # Clone model and train
            model = clone(self.model)
            trimmed_train, trimmed_y = self._apply_tomek_links(X_train, y_train)
            model.fit(
                trimmed_train, 
                trimmed_y, 
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Make predictions
            oof_pred[idx_val] = model.predict_proba(X_val)[:,1]
            test_pred += model.predict_proba(self.test)[:,1]/self.n_split
            
            # Calculate fold score
            fold_score = roc_auc_score(y_val, oof_pred[idx_val])
            scores.append(fold_score)
            print(f"Fold AUC: {fold_score}")
            
        final_score = roc_auc_score(self.target, oof_pred)
        print(f"Mean AUC: {final_score}")
        
        return oof_pred, test_pred, final_score
    
    def _apply_tomek_links(self, X, y):
        """Apply Tomek links undersampling"""
        tl = TomekLinks(sampling_strategy='auto')
        return tl.fit_resample(X, y) 