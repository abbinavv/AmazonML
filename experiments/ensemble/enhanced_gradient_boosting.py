"""
Enhanced Gradient Boosting Ensemble with Advanced Feature Extraction
Uses the sophisticated feature engineering from meta_learning_model.py
"""

import os
import re
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate SMAPE metric"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

class AdvancedFeatureExtractor:
    """Advanced feature extraction based on meta_learning_model.py"""
    
    def __init__(self):
        self._keyword_groups: Dict[str, List[str]] = {
            'qual_budget': ['budget', 'affordable', 'economy', 'basic', 'value', 'cheap', 'discount', 'clearance'],
            'qual_premium': ['premium', 'luxury', 'deluxe', 'professional', 'authentic', 'genuine', 'elite', 'superior', 'high-end'],
            'qual_artisan': ['handmade', 'artisan', 'craft', 'handcrafted', 'custom', 'bespoke', 'artisanal'],
            'cat_automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'oil', 'brake', 'automotive', 'mechanic'],
            'cat_jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'gold', 'silver', 'diamond', 'platinum', 'bracelet'],
            'cat_health': ['health', 'vitamin', 'supplement', 'organic', 'wellness', 'medical', 'therapeutic'],
            'cat_electronics': ['electronic', 'device', 'smart', 'digital', 'tech', 'wireless', 'bluetooth', 'gadget'],
            'cat_tools': ['tool', 'drill', 'hammer', 'equipment', 'hardware', 'workshop', 'construction'],
            'cat_home': ['home', 'kitchen', 'furniture', 'house', 'decor', 'living', 'household'],
            'cat_food': ['food', 'snack', 'gourmet', 'cooking', 'recipe', 'ingredient', 'culinary'],
            'brand_indicators': ['brand', 'official', 'licensed', 'certified', 'authorized', 'trademark'],
            'cat_clothing': ['clothing', 'shirt', 'dress', 'fashion', 'apparel', 'wear', 'textile'],
            'cat_sports': ['sport', 'fitness', 'exercise', 'athletic', 'gym', 'outdoor', 'recreation'],
            'cat_beauty': ['beauty', 'cosmetic', 'skincare', 'makeup', 'fragrance', 'grooming'],
            'size_indicators': ['large', 'small', 'medium', 'xl', 'xs', 'big', 'mini', 'jumbo', 'compact'],
            'material_metal': ['metal', 'steel', 'aluminum', 'brass', 'copper', 'iron', 'alloy'],
            'material_fabric': ['cotton', 'silk', 'polyester', 'wool', 'leather', 'fabric', 'textile'],
            'material_plastic': ['plastic', 'polymer', 'vinyl', 'synthetic', 'acrylic', 'resin']
        }

    def extract_features(self, catalog_content: str, image_link: str) -> Dict[str, float]:
        """Extract comprehensive features"""
        text = str(catalog_content or '')
        tl = text.lower()
        features = {}

        # Enhanced text features
        features.update(self._extract_text_features(text, tl))
        
        # Image URL features  
        features.update(self._extract_image_features(image_link))
        
        # Keyword group features
        features.update(self._extract_keyword_features(tl))
        
        return features

    def _extract_text_features(self, text: str, tl: str) -> Dict[str, float]:
        """Enhanced text feature extraction"""
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['unique_word_ratio'] = len(set(words)) / max(len(words), 1)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['punctuation_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        
        # Advanced text patterns
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['title_case_words'] = sum(1 for w in words if w.istitle()) / max(len(words), 1)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['parentheses_count'] = text.count('(') + text.count(')')
        
        # Number and measurement features
        features['number_count'] = len(re.findall(r'\d+', text))
        features['decimal_count'] = len(re.findall(r'\d+\.\d+', text))
        features['measurement_units'] = len(re.findall(r'\d+\s*(inch|ft|cm|mm|kg|lb|oz|ml|l)\b', tl))
        
        return features

    def _extract_image_features(self, image_link: str) -> Dict[str, float]:
        """Extract features from image URL"""
        features = {}
        
        if not image_link or pd.isna(image_link):
            # Default values for missing images
            features.update({
                'img_has_url': 0, 'img_url_length': 0, 'img_domain_len': 0,
                'img_path_segments': 0, 'img_file_extension_jpg': 0, 'img_file_extension_png': 0,
                'img_dimensions_in_url': 0, 'img_quality_indicators': 0, 'img_cdn_usage': 0
            })
            return features

        url_str = str(image_link)
        features['img_has_url'] = 1
        features['img_url_length'] = len(url_str)
        
        try:
            parsed = urlparse(url_str)
            features['img_domain_len'] = len(parsed.netloc)
            features['img_path_segments'] = len([p for p in parsed.path.split('/') if p])
            
            # File extension
            path_lower = parsed.path.lower()
            features['img_file_extension_jpg'] = 1 if '.jpg' in path_lower or '.jpeg' in path_lower else 0
            features['img_file_extension_png'] = 1 if '.png' in path_lower else 0
            
            # URL content analysis
            url_lower = url_str.lower()
            features['img_dimensions_in_url'] = len(re.findall(r'\d{2,4}x\d{2,4}', url_lower))
            features['img_quality_indicators'] = sum(1 for term in ['hd', 'high', 'quality', 'premium', '4k'] if term in url_lower)
            features['img_cdn_usage'] = 1 if any(cdn in parsed.netloc for cdn in ['cdn', 'cloudfront', 'amazonaws']) else 0
            
        except Exception:
            # Fallback for malformed URLs
            features.update({
                'img_domain_len': 0, 'img_path_segments': 0, 'img_file_extension_jpg': 0,
                'img_file_extension_png': 0, 'img_dimensions_in_url': 0, 'img_quality_indicators': 0,
                'img_cdn_usage': 0
            })
        
        return features

    def _extract_keyword_features(self, text_lower: str) -> Dict[str, float]:
        """Extract keyword group features"""
        features = {}
        
        for group_name, keywords in self._keyword_groups.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'kw_{group_name}'] = count
            features[f'kw_{group_name}_binary'] = 1 if count > 0 else 0
        
        return features

# --- Load and prepare data ---
print("Loading data...")
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# Initialize feature extractor
feature_extractor = AdvancedFeatureExtractor()

# Extract features
print("Extracting features...")
def extract_row_features(row):
    return feature_extractor.extract_features(row['catalog_content'], row['image_link'])

# Extract training features
X_train_features = []
for idx, row in train.iterrows():
    if idx % 10000 == 0:
        print(f"Processing training row {idx}/{len(train)}")
    X_train_features.append(extract_row_features(row))

# Extract test features
X_test_features = []
for idx, row in test.iterrows():
    if idx % 10000 == 0:
        print(f"Processing test row {idx}/{len(test)}")
    X_test_features.append(extract_row_features(row))

# Convert to DataFrames
X_train = pd.DataFrame(X_train_features)
X_test = pd.DataFrame(X_test_features)
y_train = train['price']

print(f"Feature matrix shape: {X_train.shape}")
print(f"Features: {list(X_train.columns)}")

# Handle any missing values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Definitions ---
print("Initializing models...")
xgb = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgb = LGBMRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# --- Cross-Validation Ensemble ---
print("Training ensemble with cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds_xgb = np.zeros(len(X_test_scaled))
preds_lgb = np.zeros(len(X_test_scaled))
val_preds = np.zeros(len(X_train_scaled))
val_targets = np.zeros(len(X_train_scaled))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    print(f"Training fold {fold + 1}/5...")
    
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train models
    xgb.fit(X_tr, y_tr)
    lgb.fit(X_tr, y_tr)
    
    # Validate
    xgb_val_pred = xgb.predict(X_val)
    lgb_val_pred = lgb.predict(X_val)
    val_pred = (xgb_val_pred + lgb_val_pred) / 2
    
    val_preds[val_idx] = val_pred
    val_targets[val_idx] = y_val
    
    # Test predictions
    preds_xgb += xgb.predict(X_test_scaled) / kf.n_splits
    preds_lgb += lgb.predict(X_test_scaled) / kf.n_splits
    
    # Fold SMAPE
    fold_smape = smape(y_val, val_pred)
    print(f"  Fold {fold + 1} SMAPE: {fold_smape:.4f}%")

# --- Final Results ---
cv_smape = smape(val_targets, val_preds)
print(f"\nEnhanced Gradient Boosting Ensemble SMAPE (CV): {cv_smape:.4f}%")

# Final test predictions
test_preds = (preds_xgb + preds_lgb) / 2
output = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': test_preds
})

output.to_csv('dataset/test_out_enhanced_gradient_boosting.csv', index=False)
print("Enhanced test predictions saved to dataset/test_out_enhanced_gradient_boosting.csv")

print(f"\nPrediction statistics:")
print(f"  Mean: ${test_preds.mean():.2f}")
print(f"  Median: ${np.median(test_preds):.2f}")
print(f"  Std: ${test_preds.std():.2f}")
print(f"  Range: ${test_preds.min():.2f} - ${test_preds.max():.2f}")