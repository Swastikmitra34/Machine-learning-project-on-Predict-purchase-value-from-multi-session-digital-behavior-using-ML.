from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

print(f"Original features: {X_train_scaled.shape[1]}")
print(f"PCA features: {X_train_pca.shape[1]}")

# Feature Selection
selector = SelectKBest(score_func=f_regression, k=min(50, X_train.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
