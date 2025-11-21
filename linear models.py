from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import StackingRegressor

# Define the base models (ensure they are regressors)
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1)
lgb_model = LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1)
ridge_model = Ridge(alpha=1.0)

base_models = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('ridge', ridge_model)
]

# Final model for stacking
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=LinearRegression(),
    cv=5
)

# Fit and predict
stacking_model.fit(X_train_scaled, y_train)
y_pred_stack = stacking_model.predict(X_val_scaled)
stack_r2 = r2_score(y_val, y_pred_stack)
print(f"Stacking R² Score: {stack_r2:.4f}")
