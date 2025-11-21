# Select best performing model and retrain on full training data
best_models = {
    'xgb': r2_val_xgb,
    'lgb': lgb_r2,
    'stack': stack_r2,
    'mlp': mlp_r2
}

best_model_name = 'xgb'
print(f"Best model: {best_model_name} with R² = {best_models[best_model_name]:.4f}")

# Retrain best model on full training data
if best_model_name == 'xgb':
    final_model = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
elif best_model_name == 'lgb':
    final_model = lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.05, random_state=42)
elif best_model_name == 'stack':
    final_model = stacking_model
else:
    final_model = mlp_model

# Use the SAME feature columns as validation
print(f"Using feature columns: {feature_cols}")

# Prepare final training data - use the already processed training features
X_final = train_features[feature_cols].copy()
y_final = train_df['purchaseValue'].copy()

# Apply the same numeric conversion column by column
print("Converting final training data to numeric...")

# Use apply method for robust conversion
try:
    X_final = X_final.apply(pd.to_numeric, errors='coerce')
except Exception as e:
    print(f"Bulk conversion failed: {e}")
    # Fallback method
    for col in X_final.columns:
        try:
            X_final[col] = pd.to_numeric(X_final[col].astype(str), errors='coerce')
        except Exception as col_error:
            print(f"Error with column {col}: {col_error}")

X_final = X_final.fillna(0)
print("Numeric conversion completed!")

print(f"Final training data shape: {X_final.shape}")

# Use the same scaler or create a new one
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)

print(f"Training final model: {best_model_name}")
final_model.fit(X_final_scaled, y_final)
print("Final model training completed!")
