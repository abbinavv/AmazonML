"""
Optimized Meta-Learning Model with Direct SMAPE Optimization
Target: Push SMAPE below 48% using advanced techniques:
1. Robust feature engineering
2. Direct SMAPE optimization
3. Stacking with SMAPE-optimized base models
4. Quantile regression ensemble
5. Outlier-aware training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# Removed CatBoost import due to installation issues
from sklearn.linear_model import HuberRegressor, QuantileRegressor
import re
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

def extract_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract comprehensive features with focus on price-relevant signals"""
    
    features = pd.DataFrame()
    
    # Basic numeric features
    features['content_len'] = df['catalog_content'].str.len()
    features['content_word_count'] = df['catalog_content'].str.split().str.len()
    
    # Price pattern features
    price_pattern = r'\$\d+\.?\d*|\d+\.?\d*\s*dollars?|\d+\.?\d*\s*USD'
    features['has_price'] = df['catalog_content'].str.contains(price_pattern, case=False, regex=True).astype(int)
    features['price_count'] = df['catalog_content'].str.count(price_pattern)
    
    # Brand/quality signals
    brand_words = ['authentic', 'genuine', 'official', 'original', 'brand new', 'sealed', 'rare']
    for word in brand_words:
        features[f'has_{word}'] = df['catalog_content'].str.contains(word, case=False).astype(int)
    
    # Condition signals
    condition_words = ['new', 'used', 'like new', 'excellent', 'good', 'fair', 'poor']
    for word in condition_words:
        features[f'condition_{word}'] = df['catalog_content'].str.contains(word, case=False).astype(int)
    
    # Size/quantity features
    size_pattern = r'\d+\s*(x|\*)\s*\d+'
    features['has_dimensions'] = df['catalog_content'].str.contains(size_pattern, case=False).astype(int)
    
    quantity_pattern = r'(\d+)\s*(pc|piece|pack|set)s?'
    features['has_quantity'] = df['catalog_content'].str.contains(quantity_pattern, case=False).astype(int)
    
    # Material value signals
    valuable_materials = ['gold', 'silver', 'leather', 'diamond', 'premium', 'luxury']
    for material in valuable_materials:
        features[f'has_{material}'] = df['catalog_content'].str.contains(material, case=False).astype(int)
    
    # Enhanced value signals
    value_words = ['limited edition', 'collectors', 'exclusive', 'vintage', 'antique', 'custom', 'handmade']
    for word in value_words:
        features[f'has_{word}'] = df['catalog_content'].str.contains(word, case=False).astype(int)
    
    # URL features
    def extract_url_features(url):
        try:
            parsed = urlparse(url)
            path_segments = [x for x in parsed.path.split('/') if x]
            return {
                'path_depth': len(path_segments),
                'has_query': int(bool(parsed.query)),
                'has_params': int('?' in url),
                'path_contains_size': int(bool(re.search(r'\d+x\d+', parsed.path))),
                'has_amazon_domain': int('amazon' in parsed.netloc.lower()),
                'has_cdn_domain': int('cdn' in parsed.netloc.lower()),
                'has_cloudfront': int('cloudfront' in parsed.netloc.lower()),
                'has_media': int('media' in parsed.netloc.lower())
            }
        except:
            return {
                'path_depth': 0, 'has_query': 0, 'has_params': 0, 'path_contains_size': 0,
                'has_amazon_domain': 0, 'has_cdn_domain': 0, 'has_cloudfront': 0, 'has_media': 0
            }
    
    url_features = df['image_link'].apply(extract_url_features).apply(pd.Series)
    features = pd.concat([features, url_features], axis=1)
    
    # Image quality/size features from URL
    features['is_high_res'] = df['image_link'].str.contains(r'1\d{3,}x1\d{3,}|2\d{3,}x2\d{3,}', case=False).astype(int)
    features['has_thumbnail'] = df['image_link'].str.contains(r'thumb|small|preview', case=False).astype(int)
    
    return features

def create_text_features(df: pd.DataFrame, n_components: int = 100) -> np.ndarray:
    """Create dense text features using TF-IDF and SVD"""
    
    # Advanced tokenization pattern
    token_pattern = r'(?u)\b[A-Za-z]+\b|\$\d+\.?\d*|\d+\.?\d*\s*dollars?|\d+\.?\d*\s*USD|\d+\s*x\s*\d+|#[A-Fa-f0-9]{6}'
    
    # Custom analyzer that handles both words and special patterns
    def custom_analyzer(text):
        # Extract prices
        prices = re.findall(r'\$\d+\.?\d*|\d+\.?\d*\s*dollars?|\d+\.?\d*\s*USD', text.lower())
        # Extract dimensions
        dims = re.findall(r'\d+\s*x\s*\d+', text.lower())
        # Extract regular words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        # Extract potential model numbers
        models = re.findall(r'[a-z0-9]+-?[a-z0-9]+', text.lower())
        return prices + dims + words + models
    
    # Create TF-IDF features with custom preprocessing
    tfidf = TfidfVectorizer(
        max_features=10000,
        analyzer=custom_analyzer,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9
    )
    
    text_features = tfidf.fit_transform(df['catalog_content'])
    
    # Reduce dimensionality while preserving important signals
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense_text = svd.fit_transform(text_features)
    
    return dense_text

class SMAPEOptimizedModel:
    """Base model wrapper that optimizes for SMAPE"""
    
    def __init__(self, base_model, quantile_levels=[0.1, 0.5, 0.9]):
        self.base_model = base_model
        self.quantile_models = [
            QuantileRegressor(quantile=q, alpha=0.5)
            for q in quantile_levels
        ]
        
    def fit(self, X, y):
        # Train base model
        self.base_model.fit(X, y)
        
        # Train quantile models on residuals
        base_preds = self.base_model.predict(X)
        for qmodel in self.quantile_models:
            qmodel.fit(X, y - base_preds)
            
        return self
    
    def predict(self, X):
        # Get base predictions
        base_preds = self.base_model.predict(X)
        
        # Get quantile adjustments
        adjustments = np.array([
            qmodel.predict(X) for qmodel in self.quantile_models
        ])
        
        # Combine predictions with weighted quantile adjustments
        weights = np.array([0.2, 0.6, 0.2])  # Emphasize median prediction
        final_adjustment = np.average(adjustments, axis=0, weights=weights)
        
        return base_preds + final_adjustment

def build_base_models():
    """Create SMAPE-optimized base models"""
    
    models = {
        'lgb': SMAPEOptimizedModel(
            LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                num_leaves=31,
                max_depth=6,
                min_data_in_leaf=20,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                random_state=42
            )
        ),
        'xgb': SMAPEOptimizedModel(
            XGBRegressor(
                n_estimators=1000,
                learning_rate=0.01,
                max_depth=6,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        ),
        # Removed CatBoost model due to installation issues
        'huber': SMAPEOptimizedModel(
            HuberRegressor(
                epsilon=1.35,
                max_iter=1000,
                alpha=0.0001
            )
        )
    }
    
    return models

def generate_oof_predictions(X, y, models, n_splits=5):
    """Generate out-of-fold predictions for meta-learning"""
    
    oof_predictions = np.zeros((len(X), len(models)))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Processing fold {fold + 1}/{n_splits}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for i, (name, model) in enumerate(models.items()):
            model_clone = SMAPEOptimizedModel(
                type(model.base_model)(**model.base_model.get_params())
            )
            model_clone.fit(X_train, y_train)
            oof_predictions[val_idx, i] = model_clone.predict(X_val)
    
    return oof_predictions

def train_meta_model(oof_preds, y):
    """Train meta-model using SMAPE-optimized stacking"""
    
    def objective(weights):
        # Ensure weights sum to 1 and are non-negative
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        # Calculate weighted prediction
        weighted_pred = oof_preds @ weights
        return smape(y, weighted_pred)
    
    from scipy.optimize import minimize
    
    # Initialize with equal weights
    init_weights = np.ones(oof_preds.shape[1]) / oof_preds.shape[1]
    
    # Optimize weights to minimize SMAPE
    result = minimize(
        objective,
        init_weights,
        method='L-BFGS-B',
        bounds=[(0, 1)] * oof_preds.shape[1],
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )
    
    return result.x

def main():
    """Main execution"""
    print("=" * 60)
    print("OPTIMIZED META-LEARNING MODEL")
    print("Target: Push SMAPE below 48%")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    # Extract features
    print("\nExtracting features...")
    train_features = extract_advanced_features(train)
    test_features = extract_advanced_features(test)
    
    print("Creating text features...")
    train_text = create_text_features(train)
    test_text = create_text_features(test)
    
    # Combine all features
    X_train = np.hstack([train_features, train_text])
    X_test = np.hstack([test_features, test_text])
    y_train = train['price'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build and train base models
    print("\nTraining base models...")
    models = build_base_models()
    
    # Generate OOF predictions
    print("\nGenerating out-of-fold predictions...")
    oof_predictions = generate_oof_predictions(X_train, y_train, models)
    
    # Train meta-model
    print("\nTraining meta-model...")
    meta_weights = train_meta_model(oof_predictions, y_train)
    
    # Print model weights
    print("\nOptimized model weights:")
    for name, weight in zip(models.keys(), meta_weights):
        print(f"  {name}: {weight:.4f}")
    
    # Make final predictions
    print("\nGenerating final predictions...")
    test_predictions = np.zeros((len(X_test), len(models)))
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        test_predictions[:, i] = model.predict(X_test)
    
    final_predictions = test_predictions @ meta_weights
    
    # Save predictions
    output = pd.DataFrame({
        'sample_id': test['sample_id'],
        'price': final_predictions
    })
    output.to_csv('dataset/test_out_optimized_meta_learning.csv', index=False)
    
    print("\nPrediction statistics:")
    print(f"  Mean: ${final_predictions.mean():.2f}")
    print(f"  Median: ${np.median(final_predictions):.2f}")
    print(f"  Std: ${final_predictions.std():.2f}")
    print(f"  Range: ${final_predictions.min():.2f} - ${final_predictions.max():.2f}")
    
    print("\nSaved predictions to dataset/test_out_optimized_meta_learning.csv")
    print("=" * 60)
    
if __name__ == "__main__":
    main()