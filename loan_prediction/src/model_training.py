class cv_trainer():
    """
    Cross-validation trainer class for model evaluation
    """
    def __init__(self, model, train_data, target_data, test_data, n_split, model_type, cat_ft=None):
        """
        Initialize CV trainer
        
        Args:
            model: ML model instance
            train_data: Training features
            target_data: Target variable
            test_data: Test features
            n_split: Number of CV folds
            model_type: Type of model being used
            cat_ft: Categorical features list (optional)
        """
        self.cat_ft = cat_ft
        self.model = model
        self.train = train_data
        self.target = target_data
        self.test = test_data
        self.n_split = n_split
        self.model_Type = model_type
        
    def cv(self):
        """
        Perform cross-validation training
        
        Returns:
            Tuple of (out-of-fold predictions, test predictions, average score)
        """
        fold = StratifiedKFold(n_splits=self.n_split, shuffle=True, random_state=42)
        oof_pred = np.zeros(len(self.train))
        test_pred = np.zeros(len(self.test))
        score = []
        
        # Iterate through folds
        for idx_train, idx_val in fold.split(self.train, self.target):
            # Split data into training and validation sets
            X_train, X_val = self.train.iloc[idx_train], self.train.iloc[idx_val]
            y_train, y_val = self.target.iloc[idx_train], self.target.iloc[idx_val]
            
            # Clone model and apply Tomek links undersampling
            model = clone(self.model)
            trimmed_train, trimmed_y = tomek_trim(X_train, y_train)
            
            # Train model and make predictions
            model.fit(trimmed_train, trimmed_y, eval_set=[(X_val, y_val)], verbose=False)
            oof_pred[idx_val] = model.predict_proba(X_val)[:,1]
            test_pred += model.predict_proba(self.test)[:,1]/self.n_split
            
            # Calculate and store fold score
            fold_scores = roc_auc_score(y_val, oof_pred[idx_val])
            print(f"Fold AUC: {fold_scores}")
            score.append(fold_scores)
            
        # Calculate final score
        score = roc_auc_score(self.target, oof_pred)
        print(f"mean auc: {score}")
        return oof_pred, test_pred, score 