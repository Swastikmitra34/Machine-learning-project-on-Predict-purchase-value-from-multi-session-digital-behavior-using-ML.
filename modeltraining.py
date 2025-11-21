from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Handle any remaining issues with data types
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_val = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"Final X_train shape: {X_train.shape}")
print(f"Final X_train dtypes: {X_train.dtypes.unique()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Baseline Random Forest
baseline_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
baseline_model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred_baseline = baseline_model.predict(X_val_scaled)
baseline_r2 = r2_score(y_val, y_pred_baseline)
print(f"Baseline R² Score: {baseline_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': baseline_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
