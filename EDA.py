import matplotlib.pyplot as plt
import seaborn as sns 
# Basic info about the dataset
print("Missing values:")
print(train_df.isnull().sum())

# Target distribution
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(train_df['purchaseValue'], bins=50, alpha=0.7)
plt.title('Purchase Value Distribution')
plt.xlabel('Purchase Value')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(train_df['purchaseValue']), bins=50, alpha=0.7)
plt.title('Log(Purchase Value + 1) Distribution')
plt.xlabel('Log(Purchase Value + 1)')
plt.tight_layout()
plt.show()

# Correlation with numerical features
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    correlation_matrix = train_df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()
