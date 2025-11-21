from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV


def xgb_model(X_train, y_train, X_val, y_val):
    base_model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric='rmse',
        tree_method='hist'
    )

    param_dist = {
        'n_estimators': [878],
        'max_depth': [5],
        'learning_rate': [0.12979393669240552],
        'subsample': [0.7673803589287694],
        'colsample_bytree': [0.7521119734072627],
        'colsample_bylevel': [0.7586976349436075],
        'reg_alpha': [1],
        'reg_lambda': [1.230158874253127],
        'min_child_weight': [5],
        'gamma': [2]
    }

    print("🔍 Searching for optimal hyperparameters...")
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=1,  # You already fixed the params, so only 1 iteration needed
        scoring='r2',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )

    random_search.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False
    )

    best_model = random_search.best_estimator_

    y_val_pred = best_model.predict(X_val)
    y_train_pred = best_model.predict(X_train)

    val_r2 = r2_score(y_val, y_val_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    print(f"\n✅ Best XGBoost Model Performance:")
    print(f"📈 Validation R² Score:    {val_r2:.4f}")
    print(f"📉 Training R² Score:      {train_r2:.4f}")
    print(f"🔧 Best Parameters:        {random_search.best_params_}")

    return best_model, val_r2, train_r2


# Call the function (make sure X_train_scaled, y_train, X_val_scaled, y_val are defined)
xgb_best_model, r2_val_xgb, r2_train_xgb = xgb_model(X_train_scaled, y_train, X_val_scaled, y_val)

# --- LightGBM ---
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
lgb_model.fit(X_train_scaled, y_train)
y_pred_lgb = lgb_model.predict(X_val_scaled)
lgb_r2 = r2_score(y_val, y_pred_lgb)

# --- Gradient Boosting ---
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_val_scaled)
gb_r2 = r2_score(y_val, y_pred_gb)

# --- Results ---
print(f"\n📊 Final Validation R² Scores:")
print(f"XGBoost R² Score:          {r2_val_xgb:.4f}")
print(f"LightGBM R² Score:         {lgb_r2:.4f}")
print(f"Gradient Boosting R² Score:{gb_r2:.4f}")
