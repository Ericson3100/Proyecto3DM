
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
def train_decision_tree(X_train, y_train, X_val, y_val):
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    tree = DecisionTreeRegressor(random_state=42)
    grid = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print("Árbol de Decisión (Validación):")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.2f}") 
    
    return best_model, mse, r2