import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class lightGBMRegressor:
    """
    LightGBM model for market value prediction (regression).
    """

    def __init__(self, X_train=None, y_train=None, X_valid=None, y_valid=None):
        """
        Initialize the LightGBM regression model.

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
        self.model = lgb.LGBMRegressor(
            objective='regression',
            random_state=42,
            n_estimators=100
        )

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
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.9, 1.0],
            'colsample_bytree': [0.7, 1.0]
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        # Combine training and validation sets for more generalizable tuning
        X_combined = np.concatenate((self.X_train, self.X_valid), axis=0)
        y_combined = np.concatenate((self.y_train, self.y_valid), axis=0)

        grid_search.fit(X_combined, y_combined)

        return grid_search.best_params_

    def get_trained_model(self):
        """
        Train the LightGBM model and evaluate on the test set.

        Returns:
            lgb.LGBMRegressor: Trained model.
            dict: RMSE and R² scores.
        """
        self.model.set_params(n_estimators=500)
        self.model.fit(self.X_train, self.y_train)

        # Predict on test set
        y_pred = self.model.predict(self.X_valid)

        rmse = np.sqrt(mean_squared_error(self.y_valid, y_pred))
        r2 = r2_score(self.y_valid, y_pred)

        return self.model, rmse, r2
