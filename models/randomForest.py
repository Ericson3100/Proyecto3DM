from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
def train_random_forest(X_train, y_train, X_val, y_val):
    param_grid = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    forest = RandomForestRegressor(random_state=42, n_estimators=100)
    grid = GridSearchCV(forest, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
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