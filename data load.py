import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# Load data
train_df = pd.read_csv('/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv')
test_df = pd.read_csv('/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTrain columns: {train_df.columns.tolist()}")

# Debug data types
print(f"\nData types in train:")
print(train_df.dtypes)

print(f"\nTarget statistics:")
if 'purchaseValue' in train_df.columns:
    print(train_df['purchaseValue'].describe())
else:
    print("Target column 'purchaseValue' not found. Available columns:", train_df.columns.tolist())

# Check for any obvious issues
print(f"\nSample of first few rows:")
print(train_df.head())
