import pandas as pd
import numpy as np

print("Preparing test predictions...")

# STEP 1: Add ID 
test_features = test_features.reset_index(drop=True)
test_features['id'] = range(len(test_features))
print(f"Assigned unique IDs to test set. Total rows: {len(test_features)}")

# STEP 2: Extract test features for prediction ===
test_X = test_features[feature_cols].copy()

#STEP 3: Scale 
print(" Scaling test features...")
X_test_scaled = scaler_final.transform(test_X)

#STEP 4: Predict 
print("Generating predictions...")
try:
    test_predictions = final_model.predict(X_test_scaled)
except Exception as e:
    print(f" Prediction failed with final_model: {e}\n Using fallback model.")
    test_predictions = baseline_model.predict(X_test_scaled)

# STEP 5: Create submission 
submission = pd.DataFrame({
    'id': test_features['id'],
    'purchaseValue': test_predictions
})

# STEP 6: Cleanup and Save 
submission['purchaseValue'] = submission['purchaseValue'].replace([np.inf, -np.inf], 0).fillna(0)

print("\nFinal Submission Preview:")
print(submission.head())
print(submission['purchaseValue'].describe())

assert len(submission) == len(test_features), "Row count mismatch!"
assert submission['id'].is_unique, "Duplicate IDs in submission!"
assert submission['purchaseValue'].notnull().all(), "Missing prediction values!"

submission.to_csv("submission.csv", index=False)
print("\nSubmission saved to 'submission.csv'!")
