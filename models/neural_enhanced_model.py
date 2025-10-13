"""
Neural Network Enhanced Model: TabNet + MLP Integration
Target: Push SMAPE from 50.97% to <44% with deep learning
Combines existing ensemble with neural networks for non-linear patterns
"""

import re
import numpy as np
import pandas as pd
import time
from typing import Dict, List
from urllib.parse import urlparse

# Core ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

# Neural network libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    NEURAL_AVAILABLE = True
    print("✅ TensorFlow available for neural networks")
except ImportError:
    NEURAL_AVAILABLE = False
    print("⚠️ TensorFlow not available, using sklearn MLPRegressor")

RANDOM_STATE = 42

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

class AdvancedMultiModalExtractor:
    """Enhanced feature extraction with polynomial interactions"""
    
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
            'cat_clothing': ['clothing', 'shirt', 'dress', 'fashion', 'apparel', 'wear'],
            'cat_sports': ['sport', 'fitness', 'exercise', 'athletic', 'gym', 'outdoor'],
            'cat_beauty': ['beauty', 'cosmetic', 'skincare', 'makeup', 'fragrance'],
            'size_indicators': ['large', 'small', 'medium', 'xl', 'xs', 'big', 'mini', 'jumbo']
        }
        self._token_price_map = token_price_map or {}

    def extract_enhanced_features(self, catalog_content: str, image_link: str) -> Dict[str, float]:
        """Extract comprehensive features with advanced interactions"""
        text = str(catalog_content or '')
        tl = text.lower()
        features = {}

        # Enhanced text features
        features.update(self._extract_text_features(text, tl))
        
        # Enhanced image features
        features.update(self._extract_image_features(image_link))
        
        # Cross-modal features
        features.update(self._extract_cross_modal_features(text, image_link))
        
        # Advanced interaction features
        features.update(self._extract_interaction_features(features))
        
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
        
        # Enhanced number extraction
        numbers = self._extract_enhanced_numbers(text)
        features['number_count'] = len(numbers)
        features['max_number'] = max(numbers) if numbers else 0
        features['min_number'] = min(numbers) if numbers else 0
        features['number_range'] = features['max_number'] - features['min_number']
        features['number_std'] = float(np.std(numbers)) if len(numbers) > 1 else 0
        
        # Currency and price indicators
        features['has_dollar_sign'] = 1 if '$' in text else 0
        features['has_price_words'] = 1 if any(w in tl for w in ['price', 'cost', 'value', 'msrp']) else 0
        
        # Quantity extraction
        qty = self._extract_enhanced_quantity(tl)
        features['quantity'] = qty
        features['quantity_log'] = np.log1p(qty)
        features['is_bulk'] = 1 if qty > 1 else 0
        
        # Unit extraction
        units = self._extract_enhanced_units(tl)
        for unit, val in units.items():
            features[f'unit_{unit}'] = val
        
        # Category features
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
        
        # Token-level priors
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
            
            # URL structure
            features['img_url_length'] = min(len(image_link), 1000)
            features['img_domain_length'] = len(parsed.netloc)
            features['img_path_depth'] = len([x for x in parsed.path.split('/') if x])
            features['img_query_params'] = len(parsed.query.split('&')) if parsed.query else 0
            features['img_is_https'] = 1 if image_link.startswith('https') else 0
            
            # Platform detection
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
            
            # Format detection
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
            
            # Dimension extraction
            dimensions = self._extract_enhanced_dimensions(image_link)
            features['img_width'] = dimensions.get('width', 0)
            features['img_height'] = dimensions.get('height', 0)
            features['img_area'] = features['img_width'] * features['img_height']
            features['img_aspect_ratio'] = features['img_width'] / max(features['img_height'], 1)
            
            # Quality indicators
            quality_terms = ['hd', 'hq', 'high', 'quality', 'resolution', 'dpi', 'retina', 'professional']
            features['img_quality_score'] = sum(1 for term in quality_terms if term in url_lower)
            
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
        
        # Shared keywords
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

    def _extract_interaction_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create advanced interaction features"""
        interactions = {}
        
        # Text-quantity interactions
        if 'text_length' in features and 'quantity' in features:
            interactions['text_qty_interaction'] = features['text_length'] * features['quantity']
        
        # Price-quality interactions
        if 'qual_premium' in features and 'token_prior_mean' in features:
            interactions['premium_price_interaction'] = features['qual_premium'] * features['token_prior_mean']
        
        # Category-size interactions
        for cat in ['cat_electronics', 'cat_jewelry', 'cat_automotive']:
            if cat in features and 'img_area' in features:
                interactions[f'{cat}_size_interaction'] = features[cat] * np.log1p(features['img_area'])
        
        # Brand-platform interactions
        if 'brand_indicators' in features and 'img_platform_amazon' in features:
            interactions['brand_platform_interaction'] = features['brand_indicators'] * features['img_platform_amazon']
        
        return interactions

    def _extract_enhanced_numbers(self, text: str) -> List[float]:
        """Enhanced number extraction"""
        patterns = [r'\$?\d+\.?\d*', r'\d+\.\d+', r'\d+']
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    clean = re.sub(r'[^\d\.]', '', match)
                    if clean and '.' in clean:
                        val = float(clean)
                    else:
                        val = float(clean) if clean else 0
                    if 0.01 <= val <= 100000:
                        numbers.append(val)
                except (ValueError, TypeError):
                    continue
        return sorted(set(numbers))

    def _extract_enhanced_quantity(self, text_lower: str) -> float:
        """Enhanced quantity extraction"""
        patterns = [
            r'pack of (\d+)', r'(\d+)\s*pack', r'(\d+)\s*count', r'set of (\d+)',
            r'(\d+)\s*piece', r'(\d+)\s*items?', r'(\d+)\s*units?',
            r'quantity:?\s*(\d+)', r'qty:?\s*(\d+)', r'(\d+)\s*ct'
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
            'length_in': r'(\d+\.?\d*)\s*(in|inch|inches|")',
            'length_ft': r'(\d+\.?\d*)\s*(ft|feet|foot)',
            'length_cm': r'(\d+\.?\d*)\s*(cm|centimeter|centimeters)'
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
            r'(\d+)x(\d+)', r'_(\d+)_(\d+)', r'/(\d+)/(\d+)/',
            r'w(\d+)h(\d+)', r'(\d+)[_\-](\d+)\.jpg'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, url, re.IGNORECASE)
            for match in matches:
                try:
                    w, h = int(match[0]), int(match[1])
                    if 10 <= w <= 10000 and 10 <= h <= 10000:
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
            'img_quality_score': 0, 'cross_modal_consistency': 0.0, 'cross_modal_word_overlap': 0.0, 
            'cross_modal_brand_consistency': 0
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

def create_neural_models(input_dim: int, random_state: int = RANDOM_STATE):
    """Create neural network models"""
    models = {}
    
    if NEURAL_AVAILABLE:
        # TabNet-style model
        tf.random.set_seed(random_state)
        tabnet_model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        tabnet_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        models['tabnet'] = tabnet_model
        
        # MLP model
        mlp_model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        mlp_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        models['mlp'] = mlp_model
        
    else:
        # Fallback to sklearn
        from sklearn.neural_network import MLPRegressor
        
        models['mlp_sklearn'] = MLPRegressor(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=random_state
        )
    
    return models

def train_neural_enhanced_model(train_csv: str, sample_size: int = 40000, val_size: int = 10000):
    """Train the neural-enhanced model"""
    start_time = time.time()
    print(f"Loading {sample_size + val_size} samples...")
    df = pd.read_csv(train_csv, nrows=sample_size + val_size)
    
    train_df = df.iloc[:sample_size].copy()
    val_df = df.iloc[sample_size:sample_size+val_size].copy()
    
    # Build token priors
    print("Building enhanced token priors...")
    token_price_map = {}
    for _, r in train_df.iterrows():
        text = str(r.get('catalog_content', '') or '').lower()
        price = float(r['price'])
        for tok in set(re.findall(r"[a-zA-Z0-9$%\.\-]+", text)):
            if 0.1 <= price <= 2000 and len(tok) >= 2:
                token_price_map.setdefault(tok, []).append(price)
    
    token_price_map = {k: float(np.median(v)) for k, v in token_price_map.items() if len(v) >= 5}
    print(f"Built price priors for {len(token_price_map)} tokens")
    
    # Extract enhanced features
    print("Extracting neural-ready features...")
    fx = AdvancedMultiModalExtractor(token_price_map=token_price_map)
    X_features_train = fx.transform(train_df)
    X_features_val = fx.transform(val_df)
    
    print(f"Extracted {X_features_train.shape[1]} base features")
    
    # Add polynomial interactions
    print("Creating polynomial interactions...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    
    # Select most important features for polynomial expansion (to avoid explosion)
    important_cols = ['text_length', 'word_count', 'quantity', 'token_prior_mean', 'img_area', 
                     'qual_premium', 'cat_electronics', 'cat_jewelry', 'img_quality_score']
    important_cols = [col for col in important_cols if col in X_features_train.columns]
    
    if important_cols:
        X_poly_train = poly.fit_transform(X_features_train[important_cols])
        X_poly_val = poly.transform(X_features_val[important_cols])
        
        # Combine base features with polynomial
        X_combined_train = np.hstack([X_features_train.values, X_poly_train])
        X_combined_val = np.hstack([X_features_val.values, X_poly_val])
    else:
        X_combined_train = X_features_train.values
        X_combined_val = X_features_val.values
    
    print(f"Total features with interactions: {X_combined_train.shape[1]}")
    
    # Scale features for neural networks
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_combined_train)
    X_scaled_val = scaler.transform(X_combined_val)
    
    # Prepare targets
    y_train = np.log1p(train_df['price'].values)
    y_val_true = val_df['price'].values
    
    # Dynamic clipping
    q_low, q_high = np.percentile(train_df['price'].values, [0.1, 99.9])
    
    # Text processing
    print("Enhanced text vectorization...")
    vectorizer_word = TfidfVectorizer(
        analyzer='word', lowercase=True, ngram_range=(1, 3),
        min_df=3, max_df=0.9, max_features=300000, strip_accents='unicode',
        sublinear_tf=True
    )
    vectorizer_char = TfidfVectorizer(
        analyzer='char', ngram_range=(2, 6),
        min_df=3, max_df=0.95, max_features=150000,
        sublinear_tf=True
    )
    
    X_txt_train_word = vectorizer_word.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_word = vectorizer_word.transform(val_df['catalog_content'].fillna('').astype(str))
    X_txt_train_char = vectorizer_char.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_char = vectorizer_char.transform(val_df['catalog_content'].fillna('').astype(str))
    
    X_txt_train = hstack([X_txt_train_word, X_txt_train_char])
    X_txt_val = hstack([X_txt_val_word, X_txt_val_char])
    
    # Train ensemble models
    print("Training enhanced ensemble...")
    models = {}
    
    # Traditional models
    models['ridge'] = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    models['ridge'].fit(X_txt_train, y_train)
    
    models['elastic'] = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000)
    models['elastic'].fit(X_txt_train, y_train)
    
    models['histgb'] = HistGradientBoostingRegressor(
        loss='absolute_error', max_depth=8, max_iter=800,
        learning_rate=0.03, l2_regularization=0.02, random_state=RANDOM_STATE
    )
    models['histgb'].fit(X_scaled_train, y_train)
    
    models['extratrees'] = ExtraTreesRegressor(
        n_estimators=200, max_depth=12, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    models['extratrees'].fit(X_scaled_train, y_train)
    
    # SVD + Meta model
    print("Training meta models...")
    svd = TruncatedSVD(n_components=300, random_state=RANDOM_STATE)
    X_txt_train_svd = svd.fit_transform(X_txt_train)
    X_txt_val_svd = svd.transform(X_txt_val)
    
    X_dense_train = np.hstack([X_scaled_train, X_txt_train_svd])
    X_dense_val = np.hstack([X_scaled_val, X_txt_val_svd])
    
    models['meta_histgb'] = HistGradientBoostingRegressor(
        loss='absolute_error', max_depth=8, max_iter=1000,
        learning_rate=0.025, l2_regularization=0.015, random_state=RANDOM_STATE
    )
    models['meta_histgb'].fit(X_dense_train, y_train)
    
    # Neural networks
    print("Training neural networks...")
    neural_models = create_neural_models(X_dense_train.shape[1])
    
    if NEURAL_AVAILABLE:
        # Train TabNet
        early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        neural_models['tabnet'].fit(
            X_dense_train, y_train,
            epochs=100, batch_size=512, verbose=0,
            validation_split=0.1, callbacks=[early_stopping]
        )
        
        # Train MLP
        neural_models['mlp'].fit(
            X_dense_train, y_train,
            epochs=80, batch_size=256, verbose=0,
            validation_split=0.1, callbacks=[early_stopping]
        )
        
        models.update(neural_models)
    else:
        # Train sklearn MLP
        neural_models['mlp_sklearn'].fit(X_dense_train, y_train)
        models.update(neural_models)
    
    # Get predictions for ensemble optimization
    print("Optimizing neural-enhanced ensemble...")
    predictions = {}
    
    # Traditional model predictions
    predictions['ridge'] = models['ridge'].predict(X_txt_val)
    predictions['elastic'] = models['elastic'].predict(X_txt_val)
    predictions['histgb'] = models['histgb'].predict(X_scaled_val)
    predictions['extratrees'] = models['extratrees'].predict(X_scaled_val)
    predictions['meta_histgb'] = models['meta_histgb'].predict(X_dense_val)
    
    # Neural predictions
    if NEURAL_AVAILABLE:
        predictions['tabnet'] = models['tabnet'].predict(X_dense_val, verbose=0).flatten()
        predictions['mlp'] = models['mlp'].predict(X_dense_val, verbose=0).flatten()
    else:
        predictions['mlp_sklearn'] = models['mlp_sklearn'].predict(X_dense_val)
    
    # Optimize ensemble weights
    best_weights = None
    best_smape = float('inf')
    best_pred = None
    
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    print(f"Searching optimal weights for {n_models} models...")
    
    # Grid search for weights
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    search_count = 0
    
    # For efficiency, use coarser grid with more models
    for w1 in weight_options[::2]:
        for w2 in weight_options[::2]:
            for w3 in weight_options[::2]:
                for w4 in weight_options[::2]:
                    for w5 in weight_options[::2]:
                        remaining_weight = 1.0 - (w1 + w2 + w3 + w4 + w5)
                        if remaining_weight < 0:
                            continue
                        
                        # Distribute remaining weight among remaining models
                        if n_models > 5:
                            remaining_models = n_models - 5
                            w_remaining = remaining_weight / remaining_models
                            weights = [w1, w2, w3, w4, w5] + [w_remaining] * remaining_models
                        else:
                            weights = [w1, w2, w3, w4, w5][:n_models]
                            if sum(weights) > 1:
                                continue
                        
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
    
    print("=== Neural Enhanced Model Results ===")
    weight_str = ", ".join([f"{name}:{w:.2f}" for name, w in zip(model_names, best_weights)])
    print(f"Optimal weights: {weight_str}")
    print(f"SMAPE: {s:.2f}%")
    print(f"Correlation: {corr:.4f}")
    print(f"Pred mean: ${pred_price.mean():.2f}")
    print(f"Actual mean: ${y_val_true.mean():.2f}")
    print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    artifacts = {
        'fx': fx, 'poly': poly, 'scaler': scaler, 'important_cols': important_cols,
        'vectorizer_word': vectorizer_word, 'vectorizer_char': vectorizer_char,
        'svd': svd, 'models': models, 'model_names': model_names,
        'weights': best_weights, 'clip_low': float(q_low), 'clip_high': float(q_high)
    }
    
    return s, artifacts

def predict_neural_test(test_csv: str, artifacts: dict, out_path: str):
    """Generate test predictions using neural-enhanced model"""
    print("Generating neural-enhanced test predictions...")
    test_df = pd.read_csv(test_csv)
    
    # Extract features
    fx = artifacts['fx']
    X_features_test = fx.transform(test_df)
    
    # Add polynomial interactions
    poly = artifacts['poly']
    important_cols = artifacts['important_cols']
    
    if important_cols:
        X_poly_test = poly.transform(X_features_test[important_cols])
        X_combined_test = np.hstack([X_features_test.values, X_poly_test])
    else:
        X_combined_test = X_features_test.values
    
    # Scale features
    X_scaled_test = artifacts['scaler'].transform(X_combined_test)
    
    # Text processing
    vectorizer_word = artifacts['vectorizer_word']
    vectorizer_char = artifacts['vectorizer_char']
    X_txt_test_word = vectorizer_word.transform(test_df['catalog_content'].fillna('').astype(str))
    X_txt_test_char = vectorizer_char.transform(test_df['catalog_content'].fillna('').astype(str))
    X_txt_test = hstack([X_txt_test_word, X_txt_test_char])
    
    # SVD transform
    X_txt_test_svd = artifacts['svd'].transform(X_txt_test)
    X_dense_test = np.hstack([X_scaled_test, X_txt_test_svd])
    
    # Get all model predictions
    models = artifacts['models']
    model_names = artifacts['model_names']
    predictions = {}
    
    # Traditional models
    if 'ridge' in models:
        predictions['ridge'] = models['ridge'].predict(X_txt_test)
    if 'elastic' in models:
        predictions['elastic'] = models['elastic'].predict(X_txt_test)
    if 'histgb' in models:
        predictions['histgb'] = models['histgb'].predict(X_scaled_test)
    if 'extratrees' in models:
        predictions['extratrees'] = models['extratrees'].predict(X_scaled_test)
    if 'meta_histgb' in models:
        predictions['meta_histgb'] = models['meta_histgb'].predict(X_dense_test)
    
    # Neural models
    if 'tabnet' in models:
        predictions['tabnet'] = models['tabnet'].predict(X_dense_test, verbose=0).flatten()
    if 'mlp' in models:
        predictions['mlp'] = models['mlp'].predict(X_dense_test, verbose=0).flatten()
    if 'mlp_sklearn' in models:
        predictions['mlp_sklearn'] = models['mlp_sklearn'].predict(X_dense_test)
    
    # Weighted ensemble
    weights = artifacts['weights']
    pred_log_blend = sum(w * predictions[model] for w, model in zip(weights, model_names))
    pred_price = np.expm1(pred_log_blend)
    pred_price = np.clip(pred_price, artifacts['clip_low'], artifacts['clip_high'])
    
    # Save predictions
    out = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': np.round(pred_price, 2)})
    out.to_csv(out_path, index=False)
    print(f"Neural-enhanced predictions saved to {out_path}")

if __name__ == "__main__":
    print("🧠 Neural Network Enhanced Model")
    print("TabNet + MLP + Advanced Feature Interactions")
    print("Target: Push SMAPE from 50.97% to <44%")
    print("=" * 60)
    
    smape_val, art = train_neural_enhanced_model('dataset/train.csv', sample_size=40000, val_size=10000)
    
    if smape_val < 55.0:
        predict_neural_test('dataset/test.csv', art, 'dataset/test_out_neural_enhanced.csv')
    
    print(f"\nFinal Neural Enhanced SMAPE: {smape_val:.2f}%")
    if smape_val < 44.0:
        print("🎉🎉🎉 TARGET ACHIEVED! <44% SMAPE 🎉🎉🎉")
    else:
        gap = smape_val - 44.0
        print(f"Gap to target: {gap:.2f} percentage points")
        
        if smape_val < 50.0:
            print("🚀 Significant improvement! Consider meta-learning stacking next.")
        else:
            print("📊 Consider external data integration or advanced stacking.")