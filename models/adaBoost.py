from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_adaboost(X_train, y_train, X_val, y_val):
    param_grid = {
        'learning_rate': [0.01, 0.1, 1.0],
        'loss': ['linear', 'square', 'exponential']
    }
    ada = AdaBoostRegressor(random_state=42, n_estimators=100)
    grid = GridSearchCV(ada, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    best_model.set_params(n_estimators=500)
    y_pred = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print("\nAdaBoost (Validación):")
    print(f"MSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    return best_model, rmse, r2