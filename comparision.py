import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Dictionary of fitted models (not R² scores!)
# Make sure you've trained and stored them properly
trained_models = {
    'Baseline RF': baseline_model,
    'Ridge': ridge_model,
    'Lasso': lasso_model,
    'SGD': sgd_modelimport r2_score,
    'KNN': knn_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
    'Gradient Boosting': gb_model,
    'Stacking': stacking_model,
    'MLP': mlp_model
}

#fitting models 
for name, model in trained_models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        print(f"correct {name} R² Score: {score:.4f}")
    except Exception as e:
        print(f"wrong , Error evaluating {name}: {e}")



# Evaluation
model_scores = []
best_model = None
best_r2 = -float('inf')

print("=== Evaluating Models ===")
for name, model in trained_models.items():
    try:
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_val = r2_score(y_val, y_val_pred)
        gap = abs(r2_train - r2_val)

        model_scores.append({
            'Model': name,
            'Train R²': round(r2_train, 4),
            'Validation R²': round(r2_val, 4),
            'Overfit Gap': round(gap, 4)
        })
    except Exception as e:
          print(f"wrong, Error evaluating {name}: {e}")

# Display result
score_df = pd.DataFrame(model_scores).sort_values(by='Validation R²', ascending=False)

print("\n=== All Model Scores ===")
print(score_df.to_string(index=False))

best_model_name = score_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
print(f"\nRelaxed criteria: Using best R² model — {best_model_name} (Validation R²: {score_df.iloc[0]['Validation R²']})")
