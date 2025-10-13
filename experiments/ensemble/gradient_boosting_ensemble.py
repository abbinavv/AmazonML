"""
Gradient Boosting Ensemble for Product Price Prediction
Uses XGBoost, LightGBM, CatBoost with text and image features
"""

import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor  # Skipping due to build issues
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# --- Feature Extraction Utilities ---
def extract_features(row, image_feature_dict=None):
    """Extract comprehensive features from text and optionally images"""
    text = str(row['catalog_content']) if pd.notnull(row['catalog_content']) else ""
    
    # Basic text features
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'char_count': len(text),
        'avg_word_length': len(text) / max(len(text.split()), 1),
    }
    
    # Premium indicators
    text_lower = text.lower()
    premium_words = ['premium', 'luxury', 'deluxe', 'professional', 'authentic', 'original']
    features['premium_score'] = sum(1 for word in premium_words if word in text_lower)
    
    # Material quality indicators
    materials = ['gold', 'silver', 'platinum', 'diamond', 'leather', 'steel', 'titanium']
    features['material_score'] = sum(1 for material in materials if material in text_lower)
    
    # Category detection
    categories = {
        'jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'gold', 'silver', 'diamond', 'bracelet'],
        'automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'brake', 'oil', 'filter'],
        'tools': ['tool', 'drill', 'hammer', 'equipment', 'hardware', 'wrench'],
        'electronics': ['electronic', 'device', 'smart', 'bluetooth', 'digital', 'tech'],
        'health': ['health', 'vitamin', 'supplement', 'medical', 'organic', 'wellness'],
        'clothing': ['clothing', 'shirt', 'dress', 'fashion', 'fabric', 'apparel'],
        'home': ['home', 'kitchen', 'furniture', 'decor', 'house'],
        'food': ['food', 'snack', 'gourmet', 'cooking', 'organic', 'ingredient']
    }
    
    for category, keywords in categories.items():
        features[f'category_{category}'] = sum(1 for keyword in keywords if keyword in text_lower)
    
    # Brand indicators
    brand_words = ['brand', 'official', '®', '™', 'certified']
    features['brand_score'] = sum(1 for word in brand_words if word in text_lower)
    
    # Image features (if available)
    if image_feature_dict is not None and row['sample_id'] in image_feature_dict:
        features.update(image_feature_dict[row['sample_id']])
    
    return features

# --- Load Data ---
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# --- (Optional) Load Precomputed Image Features ---
# image_feature_dict = ... # Load from file if available
image_feature_dict = None  # Set to None if not using image features

# --- Feature Matrix Construction ---
X_train = pd.DataFrame([extract_features(row, image_feature_dict) for _, row in train.iterrows()])
y_train = train['price']
X_test = pd.DataFrame([extract_features(row, image_feature_dict) for _, row in test.iterrows()])

# --- Model Definitions ---
xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
lgb = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
# cat = CatBoostRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=0)  # Skipping

# --- Cross-Validation Ensemble ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds_xgb = np.zeros(len(X_test))
preds_lgb = np.zeros(len(X_test))
# preds_cat = np.zeros(len(X_test))  # Skipping CatBoost
val_preds = np.zeros(len(X_train))
val_targets = np.zeros(len(X_train))

for train_idx, val_idx in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    xgb.fit(X_tr, y_tr)
    lgb.fit(X_tr, y_tr)
    # cat.fit(X_tr, y_tr)  # Skipping CatBoost
    
    val_pred = (xgb.predict(X_val) + lgb.predict(X_val)) / 2  # Only XGBoost + LightGBM
    val_preds[val_idx] = val_pred
    val_targets[val_idx] = y_val
    
    preds_xgb += xgb.predict(X_test) / kf.n_splits
    preds_lgb += lgb.predict(X_test) / kf.n_splits
    # preds_cat += cat.predict(X_test) / kf.n_splits  # Skipping CatBoost

# --- SMAPE Calculation ---
def smape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

smape_score = smape(val_targets, val_preds)
print(f"\nXGBoost + LightGBM Ensemble SMAPE (CV): {smape_score:.4f}%")

# --- Final Test Predictions ---
test_preds = (preds_xgb + preds_lgb) / 2  # Only XGBoost + LightGBM
output = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': test_preds
})
output.to_csv('dataset/test_out_gradient_boosting.csv', index=False)
print("Test predictions saved to dataset/test_out_gradient_boosting.csv")
