from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

def create_features(df, encoders=None, is_train=True):
    df_features = df.copy()
    
    # Separate numerical and categorical columns
    numerical_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_features.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target and id columns from feature lists
    if 'purchaseValue' in numerical_cols:
        numerical_cols.remove('purchaseValue')
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    if 'id' in numerical_cols:
        numerical_cols.remove('id')
    
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Initialize encoders dictionary if training
    if encoders is None:
        encoders = {}
    
    # Process categorical variables
    for col in categorical_cols:
        print(f"Processing categorical column: {col}")
        
        if is_train:
            # Training phase: create and fit encoder
            # Convert all values to string and handle any remaining nulls
            df_features[col] = df_features[col].astype(str).fillna('missing')
            
            # Create label encoder
            le = LabelEncoder()
            df_features[col + '_encoded'] = le.fit_transform(df_features[col])
            encoders[col] = le
            
            print(f"  - Encoded {len(le.classes_)} unique values")
            
        else:
            # Test phase: use existing encoder
            if col in encoders:
                le = encoders[col]
                
                # Convert to string and handle nulls
                df_features[col] = df_features[col].astype(str).fillna('missing')
                
                # Handle unseen categories
                def safe_transform(x):
                    if x in le.classes_:
                        return le.transform([x])[0]
                    else:
                        # Return the index of 'missing' or 0 if 'missing' doesn't exist
                        if 'missing' in le.classes_:
                            return le.transform(['missing'])[0]
                        else:
                            return 0
                
                df_features[col + '_encoded'] = df_features[col].apply(safe_transform)
                print(f"  - Applied existing encoder for {col}")
            else:
                # Fallback: create simple numeric encoding
                df_features[col] = df_features[col].astype(str).fillna('missing')
                unique_vals = df_features[col].unique()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                df_features[col + '_encoded'] = df_features[col].map(mapping)
                print(f"  - Created fallback encoding for {col}")
    
    # Create interaction features (only if both columns exist)
    if 'browser' in categorical_cols and 'device' in categorical_cols:
        interaction_col = 'browser_device'
        df_features[interaction_col] = (df_features['browser'].astype(str) + '_' + 
                                       df_features['device'].astype(str))
        
        if is_train:
            le_interaction = LabelEncoder()
            df_features[interaction_col + '_encoded'] = le_interaction.fit_transform(
                df_features[interaction_col]
            )
            encoders[interaction_col] = le_interaction
        else:
            if interaction_col in encoders:
                le_interaction = encoders[interaction_col]
                
                def safe_transform_interaction(x):
                    if x in le_interaction.classes_:
                        return le_interaction.transform([x])[0]
                    else:
                        return 0  # Default value for unseen interactions
                
                df_features[interaction_col + '_encoded'] = df_features[interaction_col].apply(
                    safe_transform_interaction
                )
    
    return df_features, encoders

# Apply feature engineering with error handling
try:
    print("Creating features for training data...")
    train_features, encoders = create_features(train_clean, is_train=True)
    print("Training features created successfully!")
    
    print("\nCreating features for test data...")
    test_features, _ = create_features(test_clean, encoders=encoders, is_train=False)
    print("Test features created successfully!")
    
except Exception as e:
    print(f"Error in feature creation: {e}")
    # Show available columns for debugging
    print(f"Available train columns: {train_clean.columns.tolist()}")
    print(f"Available test columns: {test_clean.columns.tolist()}")
