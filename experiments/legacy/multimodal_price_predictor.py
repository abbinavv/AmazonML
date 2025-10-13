"""
Multi-Modal Product Price Prediction: Text + Numeric + Image URL Features
Goal: Push SMAPE below 44% by combining all available modalities
Current best: 53.60% → Target: <44%
"""

import re
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

RANDOM_STATE = 42

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

class MultiModalFeatureExtractor:
    def __init__(self, token_price_map: Dict[str, float] | None = None):
        # Text feature groups
        self._keyword_groups: Dict[str, List[str]] = {
            'qual_budget': ['budget', 'affordable', 'economy', 'basic', 'value'],
            'qual_premium': ['premium', 'luxury', 'deluxe', 'professional', 'authentic', 'genuine'],
            'cat_automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'oil', 'brake'],
            'cat_jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'gold', 'silver', 'diamond'],
            'cat_health': ['health', 'vitamin', 'supplement', 'organic'],
            'cat_electronics': ['electronic', 'device', 'smart', 'digital', 'tech'],
            'cat_tools': ['tool', 'drill', 'hammer', 'equipment', 'hardware'],
            'cat_home': ['home', 'kitchen', 'furniture', 'house'],
            'cat_food': ['food', 'snack', 'gourmet', 'cooking'],
            'brand_indicators': ['brand', 'official', 'licensed', 'certified']
        }
        self._token_price_map = token_price_map or {}
        
        # Image URL patterns
        self.quality_keywords = [
            'hd', 'hq', 'high', 'quality', 'resolution', 'dpi', 'retina',
            'premium', 'professional', 'studio', 'detailed'
        ]
        self.platform_patterns = {
            'amazon': ['amazon', 'amzn', 'a.co'],
            'shopify': ['shopify', 'cdn.shopify'],
            'cloudinary': ['cloudinary', 'res.cloudinary'],
            'cdn': ['cdn', 'static', 'assets', 'media']
        }
        self.image_formats = ['jpg', 'jpeg', 'png', 'webp', 'gif']

    def extract_text_features(self, catalog_content: str) -> Dict[str, Any]:
        """Extract text-based numeric features"""
        text = str(catalog_content or '')
        tl = text.lower()
        features: Dict[str, Any] = {}

        # Quantity and size features
        qty = self._extract_quantity(tl)
        features['quantity'] = qty
        features['pack_size_log'] = float(np.log1p(qty))
        features['is_bulk'] = 1 if qty > 1 else 0

        # Text statistics
        features['text_length'] = len(text)
        wc = len(text.split())
        features['word_count'] = wc
        features['char_per_word'] = (features['text_length'] / max(wc, 1)) if wc else 0.0
        features['sentence_count'] = len(re.split(r'[.!?]+', text))

        # Number patterns
        nums = self._extract_numbers(text)
        features['number_count'] = len(nums)
        features['max_number'] = max(nums) if nums else 0.0
        features['avg_number'] = float(np.mean(nums)) if nums else 0.0

        # Category and quality keywords
        for key, words in self._keyword_groups.items():
            features[key] = self._count_keywords(tl, words)
        features['category_total'] = sum(features[k] for k in self._keyword_groups.keys() if k.startswith('cat_'))

        # Unit/size features
        unit_feats = self._extract_units(tl)
        features.update({f'u_{k}': v for k, v in unit_feats.items()})

        # Token-level price priors
        tokens = re.findall(r"[a-zA-Z0-9$%\.\-]+", tl)
        priors = [self._token_price_map[t] for t in tokens if t in self._token_price_map]
        features['token_prior_mean'] = float(np.mean(priors)) if priors else 0.0
        features['token_prior_max'] = float(np.max(priors)) if priors else 0.0
        features['token_prior_count'] = len(priors)

        return features

    def extract_image_url_features(self, image_link: str) -> Dict[str, Any]:
        """Extract features from image URL structure"""
        features = {}
        
        if pd.isna(image_link) or not isinstance(image_link, str) or len(image_link) < 10:
            return self._default_image_features()
        
        try:
            url_lower = image_link.lower()
            parsed = urlparse(image_link)
            
            # URL structure features
            features['img_url_length'] = min(len(image_link), 500)
            features['img_domain_length'] = len(parsed.netloc)
            features['img_path_segments'] = len([x for x in parsed.path.split('/') if x])
            features['img_is_https'] = 1 if image_link.startswith('https') else 0
            
            # Platform detection
            for platform, keywords in self.platform_patterns.items():
                features[f'img_platform_{platform}'] = 1 if any(kw in url_lower for kw in keywords) else 0
            
            # Image format and quality
            detected_format = None
            for fmt in self.image_formats:
                if f'.{fmt}' in url_lower:
                    features[f'img_format_{fmt}'] = 1
                    detected_format = fmt
                else:
                    features[f'img_format_{fmt}'] = 0
            
            # Format quality score
            format_quality_map = {'png': 0.9, 'jpeg': 0.8, 'jpg': 0.8, 'webp': 0.7, 'gif': 0.4}
            features['img_format_quality'] = format_quality_map.get(detected_format, 0.5)
            
            # Dimension extraction
            dimensions = self._extract_dimensions_from_url(image_link)
            features['img_width'] = dimensions.get('width', 0)
            features['img_height'] = dimensions.get('height', 0)
            features['img_max_dimension'] = max(dimensions.get('width', 0), dimensions.get('height', 0))
            features['img_has_dimensions'] = 1 if features['img_max_dimension'] > 0 else 0
            
            # Size categories
            max_dim = features['img_max_dimension']
            features['img_size_small'] = 1 if 0 < max_dim <= 300 else 0
            features['img_size_medium'] = 1 if 300 < max_dim <= 800 else 0
            features['img_size_large'] = 1 if max_dim > 800 else 0
            
            # Quality indicators
            features['img_quality_keywords'] = sum(1 for kw in self.quality_keywords if kw in url_lower)
            features['img_has_quality'] = 1 if features['img_quality_keywords'] > 0 else 0
            
            # Professional indicators
            filename = parsed.path.split('/')[-1] if parsed.path else ''
            features['img_filename_length'] = len(filename)
            features['img_uses_cdn'] = 1 if any(ind in url_lower for ind in ['cdn', 'static', 'assets']) else 0
            features['img_is_thumbnail'] = 1 if any(ind in url_lower for ind in ['thumb', 'small', 'mini']) else 0
            
            # Complexity score
            features['img_complexity'] = min(
                (features['img_path_segments'] * 0.3 + 
                 features['img_is_https'] * 0.3 + 
                 features['img_quality_keywords'] * 0.4), 3.0
            )
            
            return features
            
        except Exception:
            return self._default_image_features()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features (text + image) from dataframe"""
        rows: List[Dict[str, Any]] = []
        
        for _, row in df.iterrows():
            features = {}
            
            # Text features
            text_features = self.extract_text_features(row.get('catalog_content', ''))
            features.update(text_features)
            
            # Image URL features
            image_features = self.extract_image_url_features(row.get('image_link', ''))
            features.update(image_features)
            
            rows.append(features)
        
        return pd.DataFrame(rows).fillna(0)

    # Helper methods (same as before but with img_ prefixes for image features)
    def _extract_quantity(self, text_lower: str) -> float:
        patterns = [
            r'pack of (\d+)', r'(\d+)\s*pack', r'(\d+)\s*count', r'set of (\d+)',
            r'value:\s*(\d+\.?\d*)', r'(\d+)\s*piece', r'(\d+)\s*items?'
        ]
        for pattern in patterns:
            m = re.search(pattern, text_lower)
            if m:
                try:
                    return float(np.clip(float(m.group(1)), 1, 50))
                except Exception:
                    continue
        return 1.0

    def _extract_numbers(self, text: str) -> List[float]:
        nums = re.findall(r'\d+\.?\d*', text or '')
        return [float(x) for x in nums if 0.1 <= float(x) <= 50000]

    def _extract_units(self, text_lower: str) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        patterns = [
            ('oz', r'(\d+\.?\d*)\s*(oz|ounce)'),
            ('lb', r'(\d+\.?\d*)\s*(lb|pound)'),
            ('g', r'(\d+\.?\d*)\s*(g|gram)'),
            ('kg', r'(\d+\.?\d*)\s*(kg|kilogram)'),
            ('ml', r'(\d+\.?\d*)\s*(ml|milliliter)'),
            ('l', r'(\d+\.?\d*)\s*(l|liter)'),
            ('inch', r'(\d+\.?\d*)\s*(in|inch|")'),
            ('ft', r'(\d+\.?\d*)\s*(ft|feet)')
        ]
        for unit, pattern in patterns:
            m = re.search(pattern, text_lower)
            feats[unit] = float(m.group(1)) if m else 0.0
        return feats

    def _count_keywords(self, text_lower: str, keywords: List[str]) -> int:
        return sum(1 for keyword in keywords if keyword in text_lower)

    def _extract_dimensions_from_url(self, url: str) -> Dict[str, int]:
        dimensions = {'width': 0, 'height': 0}
        patterns = [r'(\d+)x(\d+)', r'_(\d+)_(\d+)', r'w(\d+)h(\d+)']
        for pattern in patterns:
            matches = re.findall(pattern, url)
            for match in matches:
                try:
                    w, h = int(match[0]), int(match[1])
                    if 10 <= w <= 5000 and 10 <= h <= 5000:
                        dimensions['width'] = max(dimensions['width'], w)
                        dimensions['height'] = max(dimensions['height'], h)
                except (ValueError, IndexError):
                    continue
        return dimensions

    def _default_image_features(self) -> Dict[str, float]:
        features = {
            'img_url_length': 0, 'img_domain_length': 0, 'img_path_segments': 0, 'img_is_https': 0,
            'img_width': 0, 'img_height': 0, 'img_max_dimension': 0, 'img_has_dimensions': 0,
            'img_size_small': 0, 'img_size_medium': 1, 'img_size_large': 0,
            'img_quality_keywords': 0, 'img_has_quality': 0, 'img_format_quality': 0.5,
            'img_filename_length': 0, 'img_uses_cdn': 0, 'img_is_thumbnail': 0, 'img_complexity': 0
        }
        for platform in self.platform_patterns.keys():
            features[f'img_platform_{platform}'] = 0
        for fmt in self.image_formats:
            features[f'img_format_{fmt}'] = 0
        return features

def train_multimodal_model(train_csv: str, sample_size: int = 35000, val_size: int = 8000):
    """Train comprehensive multi-modal model"""
    print(f"Loading {sample_size + val_size} rows from {train_csv}...")
    df = pd.read_csv(train_csv, nrows=sample_size + val_size)

    train_df = df.iloc[:sample_size].copy()
    val_df = df.iloc[sample_size:sample_size+val_size].copy()

    # Build token price priors
    print("Building token price priors...")
    token_price_map: Dict[str, float] = {}
    for _, r in train_df.iterrows():
        text = str(r.get('catalog_content', '') or '').lower()
        price = float(r['price'])
        for tok in set(re.findall(r"[a-zA-Z0-9$%\.\-]+", text)):
            if 0.1 <= price <= 1000:
                token_price_map.setdefault(tok, []).append(price)
    token_price_map = {k: float(np.median(v)) for k, v in token_price_map.items() if len(v) >= 3}

    print("Extracting multi-modal features...")
    fx = MultiModalFeatureExtractor(token_price_map=token_price_map)
    X_multimodal_train = fx.transform(train_df)
    X_multimodal_val = fx.transform(val_df)

    y_train = np.log1p(train_df['price'].values)
    y_val_true = val_df['price'].values
    q_low, q_high = np.percentile(train_df['price'].values, [0.5, 99.5])

    print("Vectorizing text...")
    vectorizer_word = TfidfVectorizer(
        analyzer='word', lowercase=True, ngram_range=(1, 2), 
        min_df=3, max_df=0.85, max_features=300000, strip_accents='unicode'
    )
    vectorizer_char = TfidfVectorizer(
        analyzer='char', ngram_range=(3, 5), 
        min_df=3, max_df=0.9, max_features=150000
    )
    
    X_txt_train_word = vectorizer_word.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_word = vectorizer_word.transform(val_df['catalog_content'].fillna('').astype(str))
    X_txt_train_char = vectorizer_char.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_char = vectorizer_char.transform(val_df['catalog_content'].fillna('').astype(str))

    X_txt_train = hstack([X_txt_train_word, X_txt_train_char])
    X_txt_val = hstack([X_txt_val_word, X_txt_val_char])

    print("Training models...")
    # Text models
    text_model = Ridge(alpha=1.5, random_state=RANDOM_STATE)
    text_model.fit(X_txt_train, y_train)

    robust_text_model = SGDRegressor(loss='huber', alpha=5e-5, max_iter=2000, random_state=RANDOM_STATE)
    robust_text_model.fit(X_txt_train, y_train)

    # Multi-modal numeric model
    multimodal_model = HistGradientBoostingRegressor(
        loss='absolute_error', max_depth=7, max_iter=600,
        learning_rate=0.04, l2_regularization=0.03, random_state=RANDOM_STATE
    )
    multimodal_model.fit(X_multimodal_train, y_train)

    # SVD + Meta model
    print("Training meta model...")
    svd = TruncatedSVD(n_components=300, random_state=RANDOM_STATE)
    X_txt_train_svd = svd.fit_transform(X_txt_train)
    X_txt_val_svd = svd.transform(X_txt_val)

    meta_X_train = np.hstack([X_multimodal_train.values, X_txt_train_svd])
    meta_X_val = np.hstack([X_multimodal_val.values, X_txt_val_svd])
    
    meta_model = HistGradientBoostingRegressor(
        loss='absolute_error', max_depth=7, max_iter=700,
        learning_rate=0.04, l2_regularization=0.02, random_state=RANDOM_STATE
    )
    meta_model.fit(meta_X_train, y_train)

    print("Optimizing ensemble...")
    # Get predictions
    pred_log_text = text_model.predict(X_txt_val)
    pred_log_text_robust = robust_text_model.predict(X_txt_val)
    pred_log_multimodal = multimodal_model.predict(X_multimodal_val)
    pred_log_meta = meta_model.predict(meta_X_val)

    # Extended grid search for 4-model blend
    best_weights = (0.25, 0.25, 0.25, 0.25)
    best_smape = float('inf')
    best_pred = None
    
    ws = np.linspace(0.0, 1.0, 11)  # 0.0, 0.1, ..., 1.0
    for w1 in ws:
        for w2 in ws:
            for w3 in ws:
                w4 = 1.0 - w1 - w2 - w3
                if w4 < 0:
                    continue
                pred_log_blend = (
                    w1 * pred_log_text +
                    w2 * pred_log_text_robust +
                    w3 * pred_log_multimodal +
                    w4 * pred_log_meta
                )
                pred_price_try = np.expm1(pred_log_blend)
                pred_price_try = np.clip(pred_price_try, q_low * 0.8, q_high * 1.1)
                s_try = smape(y_val_true, pred_price_try)
                if s_try < best_smape:
                    best_smape = s_try
                    best_weights = (w1, w2, w3, w4)
                    best_pred = pred_price_try

    s = best_smape
    pred_price = best_pred
    corr = np.corrcoef(y_val_true, pred_price)[0, 1]

    print("=== Multi-Modal Model Results ===")
    print(f"Weights -> text: {best_weights[0]:.3f}, robust: {best_weights[1]:.3f}, multimodal: {best_weights[2]:.3f}, meta: {best_weights[3]:.3f}")
    print(f"SMAPE: {s:.2f}%")
    print(f"Correlation: {corr:.4f}")
    print(f"Pred mean: ${pred_price.mean():.2f}")
    print(f"Actual mean: ${y_val_true.mean():.2f}")

    artifacts = {
        'fx': fx, 'vectorizer_word': vectorizer_word, 'vectorizer_char': vectorizer_char, 'svd': svd,
        'text_model': text_model, 'robust_text_model': robust_text_model, 
        'multimodal_model': multimodal_model, 'meta_model': meta_model,
        'weights': tuple(float(x) for x in best_weights),
        'clip_low': float(q_low), 'clip_high': float(q_high)
    }
    return s, artifacts

def predict_multimodal_test(test_csv: str, artifacts: dict, out_path: str):
    """Generate test predictions using multi-modal model"""
    print("Generating multi-modal test predictions...")
    test_df = pd.read_csv(test_csv)

    fx = artifacts['fx']
    X_multimodal_test = fx.transform(test_df)

    # Text processing
    vectorizer_word = artifacts['vectorizer_word']
    vectorizer_char = artifacts['vectorizer_char']
    X_txt_test_word = vectorizer_word.transform(test_df['catalog_content'].fillna('').astype(str))
    X_txt_test_char = vectorizer_char.transform(test_df['catalog_content'].fillna('').astype(str))
    X_txt_test = hstack([X_txt_test_word, X_txt_test_char])

    # Model predictions
    pred_log_text = artifacts['text_model'].predict(X_txt_test)
    pred_log_text_robust = artifacts['robust_text_model'].predict(X_txt_test)
    pred_log_multimodal = artifacts['multimodal_model'].predict(X_multimodal_test)
    
    X_txt_test_svd = artifacts['svd'].transform(X_txt_test)
    meta_X_test = np.hstack([X_multimodal_test.values, X_txt_test_svd])
    pred_log_meta = artifacts['meta_model'].predict(meta_X_test)

    # Blend
    w1, w2, w3, w4 = artifacts['weights']
    pred_log_blend = w1 * pred_log_text + w2 * pred_log_text_robust + w3 * pred_log_multimodal + w4 * pred_log_meta
    pred_price = np.expm1(pred_log_blend)
    pred_price = np.clip(pred_price, artifacts['clip_low'], artifacts['clip_high'])

    # Save
    out = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': np.round(pred_price, 2)})
    out.to_csv(out_path, index=False)
    print(f"Multi-modal predictions saved to {out_path}")

if __name__ == "__main__":
    print("Multi-Modal Product Price Prediction")
    print("Text + Numeric + Image URL Features")
    print("=" * 50)
    
    smape_val, art = train_multimodal_model('dataset/train.csv', sample_size=35000, val_size=8000)
    
    if smape_val < 55.0:
        predict_multimodal_test('dataset/test.csv', art, 'dataset/test_out_multimodal.csv')
    
    print(f"\nFinal Multi-Modal SMAPE: {smape_val:.2f}%")
    if smape_val < 44.0:
        print("🎉 TARGET ACHIEVED! <44% SMAPE")
    else:
        gap = smape_val - 44.0
        print(f"Gap to target: {gap:.2f} percentage points remaining")