from catboost import CatBoostRegressor as CatBoostModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class CatBoostRegressor:
    """
    CatBoost model for market value prediction (regression).
    """

    def __init__(self, X_train=None, y_train=None, X_valid=None, y_valid=None):
        """
        Initialize the CatBoost regression model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            X_valid (pd.DataFrame): Validation features.
            y_valid (pd.Series): Validation target.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test target.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        # Initialize the base model
        self.model = CatBoostModel(logging_level='Silent', train_dir='/tmp/catboost_temp')

        # Perform hyperparameter tuning
        self.best_params = self.optimize_hyperparameters()
        self.model.set_params(**self.best_params)

    def optimize_hyperparameters(self):
        """
        Perform hyperparameter optimization using GridSearchCV on the validation set.

        Returns:
            dict: Best hyperparameters.
        """
        param_grid = {
            'depth': [2, 4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 1],
            'l2_leaf_reg': [1, 3, 5, 7]
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3,
            n_jobs=-1,
            verbose=0
        )

        # Use combined training + validation for better generalization during grid search
        X_combined = np.concatenate((self.X_train, self.X_valid), axis=0)
        y_combined = np.concatenate((self.y_train, self.y_valid), axis=0)

        grid_search.fit(X_combined, y_combined)

        return grid_search.best_params_

    def get_trained_model(self):
        """
        Train the CatBoost model and evaluate on the validation set.

        Returns:
            CatBoostRegressor: Trained model.
            dict: MSE and R² scores.
        """
        self.model.set_params(iterations=500)  # Set a higher number of iterations for final training
        self.model.fit(self.X_train, self.y_train)

        # Predict on validation set
        y_pred = self.model.predict(self.X_valid)

        mse = mean_squared_error(self.y_valid, y_pred)
        r2 = r2_score(self.y_valid, y_pred)

        return self.model, mse, r2
