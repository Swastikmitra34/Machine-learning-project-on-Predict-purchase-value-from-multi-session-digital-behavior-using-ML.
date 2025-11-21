def clean_and_preprocess(df, is_train=True):
    df_clean = df.copy()
    
    # Handle missing values first
    for col in df_clean.columns:
        if col in ['id', 'purchaseValue']:  # Skip ID and target
            continue
            
        if df_clean[col].dtype == 'object':
            # For categorical columns, fill with most frequent value or 'unknown'
            mode_val = df_clean[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'unknown'
            df_clean[col] = df_clean[col].fillna(fill_val)
        else:
            # For numerical columns, fill with median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean

# Apply preprocessing
train_clean = clean_and_preprocess(train_df, is_train=True)
test_clean = clean_and_preprocess(test_df, is_train=False)

print("Data cleaning completed!")
print(f"Train missing values: {train_clean.isnull().sum().sum()}")
print(f"Test missing values: {test_clean.isnull().sum().sum()}")
