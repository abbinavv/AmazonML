"""
Final Optimized Model: Advanced Hyperparameter Tuning + Stacking
Push SMAPE from 51.61% towards <44% target
Last optimization step using all available techniques
"""

import re
import numpy as np
import pandas as pd
import time
from typing import Dict, List
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

class AdvancedMultiModalExtractor:
    """Enhanced feature extraction with additional signal processing"""
    
    def __init__(self, token_price_map: Dict[str, float] | None = None):
        self._keyword_groups: Dict[str, List[str]] = {
            'qual_budget': ['budget', 'affordable', 'economy', 'basic', 'value', 'cheap', 'discount'],
            'qual_premium': ['premium', 'luxury', 'deluxe', 'professional', 'authentic', 'genuine', 'elite', 'superior'],
            'cat_automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'oil', 'brake', 'automotive'],
            'cat_jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'gold', 'silver', 'diamond', 'platinum'],
            'cat_health': ['health', 'vitamin', 'supplement', 'organic', 'wellness', 'medical'],
            'cat_electronics': ['electronic', 'device', 'smart', 'digital', 'tech', 'wireless', 'bluetooth'],
            'cat_tools': ['tool', 'drill', 'hammer', 'equipment', 'hardware', 'workshop'],
            'cat_home': ['home', 'kitchen', 'furniture', 'house', 'decor', 'living'],
            'cat_food': ['food', 'snack', 'gourmet', 'cooking', 'recipe', 'ingredient'],
            'brand_indicators': ['brand', 'official', 'licensed', 'certified', 'authorized'],
            # Additional categories for better segmentation
            'cat_clothing': ['clothing', 'shirt', 'dress', 'fashion', 'apparel', 'wear'],
            'cat_sports': ['sport', 'fitness', 'exercise', 'athletic', 'gym', 'outdoor'],
            'cat_beauty': ['beauty', 'cosmetic', 'skincare', 'makeup', 'fragrance'],
            'size_indicators': ['large', 'small', 'medium', 'xl', 'xs', 'big', 'mini', 'jumbo']
        }
        self._token_price_map = token_price_map or {}

    def extract_enhanced_features(self, catalog_content: str, image_link: str) -> Dict[str, float]:
        """Extract comprehensive features with advanced signal processing"""
        text = str(catalog_content or '')
        tl = text.lower()
        features = {}

        # Enhanced text features
        features.update(self._extract_text_features(text, tl))
        
        # Enhanced image features
        features.update(self._extract_image_features(image_link))
        
        # Cross-modal features (text-image consistency)
        features.update(self._extract_cross_modal_features(text, image_link))
        
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
        
        # Capitalization patterns
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['title_case_words'] = sum(1 for w in words if w.istitle()) / max(len(words), 1)
        
        # Number extraction with enhanced patterns
        numbers = self._extract_enhanced_numbers(text)
        features['number_count'] = len(numbers)
        features['max_number'] = max(numbers) if numbers else 0
        features['min_number'] = min(numbers) if numbers else 0
        features['number_range'] = features['max_number'] - features['min_number']
        features['number_std'] = float(np.std(numbers)) if len(numbers) > 1 else 0
        
        # Currency and price indicators
        features['has_dollar_sign'] = 1 if '$' in text else 0
        features['has_price_words'] = 1 if any(w in tl for w in ['price', 'cost', 'value', 'msrp']) else 0
        
        # Quantity with enhanced patterns
        qty = self._extract_enhanced_quantity(tl)
        features['quantity'] = qty
        features['quantity_log'] = np.log1p(qty)
        features['is_bulk'] = 1 if qty > 1 else 0
        features['is_large_bulk'] = 1 if qty > 10 else 0
        
        # Enhanced unit extraction
        units = self._extract_enhanced_units(tl)
        for unit, val in units.items():
            features[f'unit_{unit}'] = val
        
        # Category features with confidence scores
        category_scores = {}
        for cat, keywords in self._keyword_groups.items():
            score = sum(1 for kw in keywords if kw in tl)
            features[cat] = score
            if cat.startswith('cat_'):
                category_scores[cat] = score
        
        # Dominant category
        if category_scores:
            dominant_cat = max(category_scores, key=category_scores.get)
            for cat in category_scores:
                features[f'{cat}_dominant'] = 1 if cat == dominant_cat else 0
        
        # Token-level priors with enhanced processing
        tokens = re.findall(r"[a-zA-Z0-9$%\.\-]+", tl)
        priors = [self._token_price_map.get(t, 0) for t in tokens if t in self._token_price_map]
        features['token_prior_mean'] = float(np.mean(priors)) if priors else 0
        features['token_prior_median'] = float(np.median(priors)) if priors else 0
        features['token_prior_max'] = float(np.max(priors)) if priors else 0
        features['token_prior_std'] = float(np.std(priors)) if len(priors) > 1 else 0
        features['token_prior_count'] = len(priors)
        features['token_coverage'] = len(priors) / max(len(tokens), 1)
        
        return features

    def _extract_image_features(self, image_link: str) -> Dict[str, float]:
        """Enhanced image URL feature extraction"""
        features = {}
        
        if pd.isna(image_link) or not isinstance(image_link, str) or len(image_link) < 10:
            return self._default_image_features()
        
        try:
            url_lower = image_link.lower()
            parsed = urlparse(image_link)
            
            # Enhanced URL structure
            features['img_url_length'] = min(len(image_link), 1000)
            features['img_domain_length'] = len(parsed.netloc)
            features['img_path_depth'] = len([x for x in parsed.path.split('/') if x])
            features['img_query_params'] = len(parsed.query.split('&')) if parsed.query else 0
            features['img_is_https'] = 1 if image_link.startswith('https') else 0
            
            # Platform detection with confidence
            platform_confidence = {}
            platforms = {
                'amazon': ['amazon', 'amzn', 'a.co'],
                'shopify': ['shopify', 'cdn.shopify'],
                'cloudinary': ['cloudinary', 'res.cloudinary'],
                'imgur': ['imgur', 'i.imgur'],
                'cdn': ['cdn', 'static', 'assets', 'media', 'images']
            }
            
            for platform, keywords in platforms.items():
                confidence = sum(1 for kw in keywords if kw in url_lower)
                features[f'img_platform_{platform}'] = confidence
                platform_confidence[platform] = confidence
            
            # Dominant platform
            if platform_confidence:
                dominant = max(platform_confidence, key=platform_confidence.get)
                for platform in platform_confidence:
                    features[f'img_{platform}_dominant'] = 1 if platform == dominant else 0
            
            # Enhanced format detection
            formats = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'svg', 'bmp']
            format_quality = {'png': 0.9, 'svg': 0.85, 'jpeg': 0.8, 'jpg': 0.8, 'webp': 0.7, 'gif': 0.4, 'bmp': 0.3}
            
            detected_formats = []
            for fmt in formats:
                if f'.{fmt}' in url_lower:
                    features[f'img_format_{fmt}'] = 1
                    detected_formats.append(fmt)
                else:
                    features[f'img_format_{fmt}'] = 0
            
            features['img_format_quality'] = max([format_quality.get(fmt, 0.5) for fmt in detected_formats], default=0.5)
            features['img_multiple_formats'] = 1 if len(detected_formats) > 1 else 0
            
            # Enhanced dimension extraction
            dimensions = self._extract_enhanced_dimensions(image_link)
            features['img_width'] = dimensions.get('width', 0)
            features['img_height'] = dimensions.get('height', 0)
            features['img_area'] = features['img_width'] * features['img_height']
            features['img_aspect_ratio'] = features['img_width'] / max(features['img_height'], 1)
            features['img_is_square'] = 1 if abs(features['img_aspect_ratio'] - 1.0) < 0.1 else 0
            features['img_is_portrait'] = 1 if features['img_aspect_ratio'] < 0.8 else 0
            features['img_is_landscape'] = 1 if features['img_aspect_ratio'] > 1.2 else 0
            
            # Size categorization
            max_dim = max(features['img_width'], features['img_height'])
            features['img_size_tiny'] = 1 if 0 < max_dim <= 150 else 0
            features['img_size_small'] = 1 if 150 < max_dim <= 400 else 0
            features['img_size_medium'] = 1 if 400 < max_dim <= 1000 else 0
            features['img_size_large'] = 1 if 1000 < max_dim <= 2000 else 0
            features['img_size_huge'] = 1 if max_dim > 2000 else 0
            
            # Quality indicators
            quality_terms = ['hd', 'hq', 'high', 'quality', 'resolution', 'dpi', 'retina', 'professional']
            features['img_quality_score'] = sum(1 for term in quality_terms if term in url_lower)
            
            # Professional indicators
            pro_terms = ['studio', 'professional', 'product', 'catalog', 'gallery']
            features['img_professional_score'] = sum(1 for term in pro_terms if term in url_lower)
            
            # Filename analysis
            filename = parsed.path.split('/')[-1] if parsed.path else ''
            features['img_filename_length'] = len(filename)
            features['img_has_hash'] = 1 if len(re.findall(r'[a-f0-9]{8,}', filename)) > 0 else 0
            features['img_has_timestamp'] = 1 if re.search(r'\d{8,}', filename) else 0
            
            return features
            
        except Exception:
            return self._default_image_features()

    def _extract_cross_modal_features(self, text: str, image_link: str) -> Dict[str, float]:
        """Extract cross-modal consistency features"""
        features = {}
        
        if not text or not image_link:
            return {'cross_modal_consistency': 0.0}
        
        text_lower = text.lower()
        url_lower = image_link.lower()
        
        # Shared keywords between text and URL
        text_words = set(re.findall(r'[a-zA-Z]+', text_lower))
        url_words = set(re.findall(r'[a-zA-Z]+', url_lower))
        
        if text_words and url_words:
            intersection = text_words.intersection(url_words)
            features['cross_modal_word_overlap'] = len(intersection) / min(len(text_words), len(url_words))
        else:
            features['cross_modal_word_overlap'] = 0.0
        
        # Brand consistency
        text_has_brand = any(word in text_lower for word in ['brand', 'official', 'authentic'])
        url_has_brand_indicator = any(term in url_lower for term in ['official', 'brand', 'authentic'])
        features['cross_modal_brand_consistency'] = 1 if text_has_brand == url_has_brand_indicator else 0
        
        return features

    def _extract_enhanced_numbers(self, text: str) -> List[float]:
        """Enhanced number extraction with decimal support"""
        patterns = [
            r'\$?\d+\.?\d*',  # Currency and decimals
            r'\d+\.\d+',      # Explicit decimals
            r'\d+'            # Integers
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Clean the match
                    clean = re.sub(r'[^\d\.]', '', match)
                    if clean and '.' in clean:
                        val = float(clean)
                    else:
                        val = float(clean) if clean else 0
                    
                    if 0.01 <= val <= 100000:  # Reasonable bounds
                        numbers.append(val)
                except (ValueError, TypeError):
                    continue
        
        return sorted(set(numbers))

    def _extract_enhanced_quantity(self, text_lower: str) -> float:
        """Enhanced quantity extraction with more patterns"""
        patterns = [
            r'pack of (\d+)', r'(\d+)\s*pack', r'(\d+)\s*count', r'set of (\d+)',
            r'(\d+)\s*piece', r'(\d+)\s*items?', r'(\d+)\s*units?',
            r'quantity:?\s*(\d+)', r'qty:?\s*(\d+)', r'(\d+)\s*ct',
            r'box of (\d+)', r'case of (\d+)', r'bundle of (\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return float(np.clip(float(match.group(1)), 1, 100))
                except (ValueError, IndexError):
                    continue
        
        return 1.0

    def _extract_enhanced_units(self, text_lower: str) -> Dict[str, float]:
        """Enhanced unit extraction"""
        unit_patterns = {
            'weight_oz': r'(\d+\.?\d*)\s*(oz|ounce|ounces)',
            'weight_lb': r'(\d+\.?\d*)\s*(lb|lbs|pound|pounds)',
            'weight_g': r'(\d+\.?\d*)\s*(g|gram|grams)',
            'weight_kg': r'(\d+\.?\d*)\s*(kg|kilogram|kilograms)',
            'volume_ml': r'(\d+\.?\d*)\s*(ml|milliliter|milliliters)',
            'volume_l': r'(\d+\.?\d*)\s*(l|liter|liters)',
            'volume_fl_oz': r'(\d+\.?\d*)\s*(fl\s*oz|fluid\s*ounce)',
            'length_in': r'(\d+\.?\d*)\s*(in|inch|inches|")',
            'length_ft': r'(\d+\.?\d*)\s*(ft|feet|foot)',
            'length_cm': r'(\d+\.?\d*)\s*(cm|centimeter|centimeters)',
            'length_mm': r'(\d+\.?\d*)\s*(mm|millimeter|millimeters)',
            'area_sqft': r'(\d+\.?\d*)\s*(sq\s*ft|square\s*feet)',
            'time_hr': r'(\d+\.?\d*)\s*(hr|hour|hours)',
            'time_min': r'(\d+\.?\d*)\s*(min|minute|minutes)'
        }
        
        units = {}
        for unit_name, pattern in unit_patterns.items():
            match = re.search(pattern, text_lower)
            try:
                units[unit_name] = float(match.group(1)) if match else 0.0
            except (ValueError, IndexError):
                units[unit_name] = 0.0
        
        return units

    def _extract_enhanced_dimensions(self, url: str) -> Dict[str, int]:
        """Enhanced dimension extraction from URLs"""
        dimensions = {'width': 0, 'height': 0}
        
        patterns = [
            r'(\d+)x(\d+)',                    # 800x600
            r'_(\d+)_(\d+)',                   # _800_600
            r'/(\d+)/(\d+)/',                  # /800/600/
            r'w(\d+)h(\d+)',                   # w800h600
            r'width[_\-]?(\d+)[_\-]?height[_\-]?(\d+)',  # width800height600
            r'(\d+)[_\-](\d+)\.jpg',          # 800_600.jpg
            r'(\d+)[x\-_](\d+)\.',            # 800x600. or 800-600. or 800_600.
            r'size[_\-]?(\d+)[x_\-](\d+)',    # size800x600
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, url, re.IGNORECASE)
            for match in matches:
                try:
                    w, h = int(match[0]), int(match[1])
                    if 10 <= w <= 10000 and 10 <= h <= 10000:  # Reasonable bounds
                        dimensions['width'] = max(dimensions['width'], w)
                        dimensions['height'] = max(dimensions['height'], h)
                except (ValueError, IndexError):
                    continue
        
        return dimensions

    def _default_image_features(self) -> Dict[str, float]:
        """Default image features"""
        return {
            'img_url_length': 0, 'img_domain_length': 0, 'img_path_depth': 0, 'img_query_params': 0,
            'img_is_https': 0, 'img_width': 300, 'img_height': 300, 'img_area': 90000, 'img_aspect_ratio': 1.0,
            'img_is_square': 1, 'img_is_portrait': 0, 'img_is_landscape': 0,
            'img_size_tiny': 0, 'img_size_small': 0, 'img_size_medium': 1, 'img_size_large': 0, 'img_size_huge': 0,
            'img_quality_score': 0, 'img_professional_score': 0, 'img_format_quality': 0.5,
            'img_filename_length': 0, 'img_has_hash': 0, 'img_has_timestamp': 0, 'img_multiple_formats': 0,
            'cross_modal_consistency': 0.0, 'cross_modal_word_overlap': 0.0, 'cross_modal_brand_consistency': 0
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe to feature matrix"""
        features_list = []
        
        for _, row in df.iterrows():
            features = self.extract_enhanced_features(
                row.get('catalog_content', ''),
                row.get('image_link', '')
            )
            features_list.append(features)
        
        return pd.DataFrame(features_list).fillna(0)

def train_ultimate_model(train_csv: str, sample_size: int = 40000, val_size: int = 10000):
    """Train the ultimate optimized model"""
    start_time = time.time()
    print(f"Loading {sample_size + val_size} samples...")
    df = pd.read_csv(train_csv, nrows=sample_size + val_size)
    
    train_df = df.iloc[:sample_size].copy()
    val_df = df.iloc[sample_size:sample_size+val_size].copy()
    
    # Build enhanced token priors
    print("Building enhanced token priors...")
    token_price_map = {}
    for _, r in train_df.iterrows():
        text = str(r.get('catalog_content', '') or '').lower()
        price = float(r['price'])
        for tok in set(re.findall(r"[a-zA-Z0-9$%\.\-]+", text)):
            if 0.1 <= price <= 2000 and len(tok) >= 2:  # Filter short tokens
                token_price_map.setdefault(tok, []).append(price)
    
    # Use median for stability, require more samples for reliability
    token_price_map = {k: float(np.median(v)) for k, v in token_price_map.items() if len(v) >= 5}
    print(f"Built price priors for {len(token_price_map)} tokens")
    
    # Extract enhanced features
    print("Extracting enhanced multi-modal features...")
    fx = AdvancedMultiModalExtractor(token_price_map=token_price_map)
    X_features_train = fx.transform(train_df)
    X_features_val = fx.transform(val_df)
    
    print(f"Extracted {X_features_train.shape[1]} features")
    
    # Prepare targets with log transform
    y_train = np.log1p(train_df['price'].values)
    y_val_true = val_df['price'].values
    
    # Dynamic clipping bounds
    q_low, q_high = np.percentile(train_df['price'].values, [0.1, 99.9])
    
    # Enhanced text processing
    print("Enhanced text vectorization...")
    vectorizer_word = TfidfVectorizer(
        analyzer='word', lowercase=True, ngram_range=(1, 3),  # Include trigrams
        min_df=3, max_df=0.9, max_features=400000, strip_accents='unicode',
        sublinear_tf=True  # Sublinear TF scaling
    )
    vectorizer_char = TfidfVectorizer(
        analyzer='char', ngram_range=(2, 6),  # Extended char range
        min_df=3, max_df=0.95, max_features=200000,
        sublinear_tf=True
    )
    
    X_txt_train_word = vectorizer_word.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_word = vectorizer_word.transform(val_df['catalog_content'].fillna('').astype(str))
    X_txt_train_char = vectorizer_char.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_char = vectorizer_char.transform(val_df['catalog_content'].fillna('').astype(str))
    
    X_txt_train = hstack([X_txt_train_word, X_txt_train_char])
    X_txt_val = hstack([X_txt_val_word, X_txt_val_char])
    
    # Train ensemble of diverse models
    print("Training diverse model ensemble...")
    
    models = {}
    
    # Text models with different algorithms
    models['ridge'] = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    models['ridge'].fit(X_txt_train, y_train)
    
    models['elastic'] = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000)
    models['elastic'].fit(X_txt_train, y_train)
    
    models['sgd'] = SGDRegressor(loss='huber', alpha=1e-5, max_iter=2000, random_state=RANDOM_STATE)
    models['sgd'].fit(X_txt_train, y_train)
    
    # Feature-based models
    models['histgb'] = HistGradientBoostingRegressor(
        loss='absolute_error', max_depth=8, max_iter=800,
        learning_rate=0.03, l2_regularization=0.02, random_state=RANDOM_STATE
    )
    models['histgb'].fit(X_features_train, y_train)
    
    models['extratrees'] = ExtraTreesRegressor(
        n_estimators=200, max_depth=12, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    models['extratrees'].fit(X_features_train, y_train)
    
    # SVD + Meta model
    print("Training meta models...")
    svd = TruncatedSVD(n_components=350, random_state=RANDOM_STATE)
    X_txt_train_svd = svd.fit_transform(X_txt_train)
    X_txt_val_svd = svd.transform(X_txt_val)
    
    # Combined dense features
    X_dense_train = np.hstack([X_features_train.values, X_txt_train_svd])
    X_dense_val = np.hstack([X_features_val.values, X_txt_val_svd])
    
    models['meta_histgb'] = HistGradientBoostingRegressor(
        loss='absolute_error', max_depth=8, max_iter=1000,
        learning_rate=0.025, l2_regularization=0.015, random_state=RANDOM_STATE
    )
    models['meta_histgb'].fit(X_dense_train, y_train)
    
    models['meta_extratrees'] = ExtraTreesRegressor(
        n_estimators=300, max_depth=15, min_samples_split=3,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    models['meta_extratrees'].fit(X_dense_train, y_train)
    
    # Get predictions for ensemble optimization
    print("Optimizing 7-model ensemble...")
    predictions = {}
    
    # Text model predictions
    predictions['ridge'] = models['ridge'].predict(X_txt_val)
    predictions['elastic'] = models['elastic'].predict(X_txt_val)
    predictions['sgd'] = models['sgd'].predict(X_txt_val)
    
    # Feature model predictions
    predictions['histgb'] = models['histgb'].predict(X_features_val)
    predictions['extratrees'] = models['extratrees'].predict(X_features_val)
    
    # Meta model predictions
    predictions['meta_histgb'] = models['meta_histgb'].predict(X_dense_val)
    predictions['meta_extratrees'] = models['meta_extratrees'].predict(X_dense_val)
    
    # Exhaustive blend optimization for 7 models
    best_weights = None
    best_smape = float('inf')
    best_pred = None
    
    # Systematic search with fewer steps due to 7 models
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    model_names = list(predictions.keys())
    
    print("Searching optimal blend weights...")
    search_count = 0
    for w1 in weight_options[::2]:  # Reduce search space
        for w2 in weight_options[::2]:
            for w3 in weight_options[::2]:
                for w4 in weight_options[::2]:
                    for w5 in weight_options[::2]:
                        for w6 in weight_options[::2]:
                            w7 = 1.0 - (w1 + w2 + w3 + w4 + w5 + w6)
                            if w7 < 0 or w7 > 1:
                                continue
                            
                            weights = [w1, w2, w3, w4, w5, w6, w7]
                            
                            # Weighted combination
                            pred_log_blend = sum(w * predictions[model] for w, model in zip(weights, model_names))
                            pred_price_try = np.expm1(pred_log_blend)
                            pred_price_try = np.clip(pred_price_try, q_low * 0.7, q_high * 1.2)
                            
                            s_try = smape(y_val_true, pred_price_try)
                            search_count += 1
                            
                            if s_try < best_smape:
                                best_smape = s_try
                                best_weights = weights
                                best_pred = pred_price_try
                                print(f"  New best SMAPE: {s_try:.3f}% (search #{search_count})")
    
    print(f"Completed {search_count} weight combinations")
    
    s = best_smape
    pred_price = best_pred
    corr = np.corrcoef(y_val_true, pred_price)[0, 1]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=== Ultimate Model Results ===")
    weight_str = ", ".join([f"{name}:{w:.2f}" for name, w in zip(model_names, best_weights)])
    print(f"Optimal weights: {weight_str}")
    print(f"SMAPE: {s:.2f}%")
    print(f"Correlation: {corr:.4f}")
    print(f"Pred mean: ${pred_price.mean():.2f}")
    print(f"Actual mean: ${y_val_true.mean():.2f}")
    print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    artifacts = {
        'fx': fx, 'vectorizer_word': vectorizer_word, 'vectorizer_char': vectorizer_char,
        'svd': svd, 'models': models, 'model_names': model_names,
        'weights': best_weights, 'clip_low': float(q_low), 'clip_high': float(q_high)
    }
    
    return s, artifacts

def predict_ultimate_test(test_csv: str, artifacts: dict, out_path: str):
    """Generate test predictions using ultimate model"""
    print("Generating ultimate test predictions...")
    test_df = pd.read_csv(test_csv)
    
    # Extract features
    fx = artifacts['fx']
    X_features_test = fx.transform(test_df)
    
    # Text processing
    vectorizer_word = artifacts['vectorizer_word']
    vectorizer_char = artifacts['vectorizer_char']
    X_txt_test_word = vectorizer_word.transform(test_df['catalog_content'].fillna('').astype(str))
    X_txt_test_char = vectorizer_char.transform(test_df['catalog_content'].fillna('').astype(str))
    X_txt_test = hstack([X_txt_test_word, X_txt_test_char])
    
    # SVD transform
    X_txt_test_svd = artifacts['svd'].transform(X_txt_test)
    X_dense_test = np.hstack([X_features_test.values, X_txt_test_svd])
    
    # Get all model predictions
    models = artifacts['models']
    model_names = artifacts['model_names']
    predictions = {}
    
    predictions['ridge'] = models['ridge'].predict(X_txt_test)
    predictions['elastic'] = models['elastic'].predict(X_txt_test)
    predictions['sgd'] = models['sgd'].predict(X_txt_test)
    predictions['histgb'] = models['histgb'].predict(X_features_test)
    predictions['extratrees'] = models['extratrees'].predict(X_features_test)
    predictions['meta_histgb'] = models['meta_histgb'].predict(X_dense_test)
    predictions['meta_extratrees'] = models['meta_extratrees'].predict(X_dense_test)
    
    # Weighted ensemble
    weights = artifacts['weights']
    pred_log_blend = sum(w * predictions[model] for w, model in zip(weights, model_names))
    pred_price = np.expm1(pred_log_blend)
    pred_price = np.clip(pred_price, artifacts['clip_low'], artifacts['clip_high'])
    
    # Save predictions
    out = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': np.round(pred_price, 2)})
    out.to_csv(out_path, index=False)
    print(f"Ultimate predictions saved to {out_path}")

if __name__ == "__main__":
    print("Ultimate Optimized Multi-Modal Model")
    print("Advanced Hyperparameter Tuning + 7-Model Ensemble")
    print("=" * 60)
    
    smape_val, art = train_ultimate_model('dataset/train.csv', sample_size=40000, val_size=10000)
    
    if smape_val < 55.0:
        predict_ultimate_test('dataset/test.csv', art, 'dataset/test_out_ultimate.csv')
    
    print(f"\nFinal Ultimate SMAPE: {smape_val:.2f}%")
    if smape_val < 44.0:
        print("🎉🎉 TARGET ACHIEVED! <44% SMAPE 🎉🎉")
    else:
        gap = smape_val - 44.0
        print(f"Gap to target: {gap:.2f} percentage points")
        print("Additional techniques: Neural networks, external data, advanced stacking")