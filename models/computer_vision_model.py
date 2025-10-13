"""
Computer Vision Enhanced Model: Actual Image Analysis
Target: Push SMAPE from 50.12% to <44% using real image features
Downloads and analyzes actual images from dataset URLs (safe within rules)
Extracts visual features: colors, quality, objects, aesthetics
"""

import re
import numpy as np
import pandas as pd
import time
import requests
from typing import Dict, List, Tuple
from urllib.parse import urlparse
import io
import os
from PIL import Image, ImageStat, ImageFilter
import cv2
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from scipy.sparse import hstack

RANDOM_STATE = 42

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

class ComputerVisionExtractor:
    """Extract features from actual downloaded images"""
    
    def __init__(self, cache_dir: str = "image_cache", timeout: int = 10):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"✅ Computer Vision Extractor initialized")
        print(f"📁 Cache directory: {cache_dir}")
        print(f"⏱️ Download timeout: {timeout}s")

    def download_image(self, image_url: str, sample_id: str) -> Image.Image | None:
        """Download and return PIL Image, with caching"""
        if pd.isna(image_url) or not isinstance(image_url, str):
            return None
            
        # Create cache filename
        cache_file = os.path.join(self.cache_dir, f"{sample_id}.jpg")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                return Image.open(cache_file).convert('RGB')
            except Exception:
                os.remove(cache_file)  # Remove corrupted cache
        
        # Download image
        try:
            response = self.session.get(image_url, timeout=self.timeout, stream=True)
            if response.status_code == 200:
                # Save to cache
                with open(cache_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Return PIL Image
                return Image.open(cache_file).convert('RGB')
        except Exception as e:
            pass  # Silently handle download failures
            
        return None

    def extract_color_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract color-based features"""
        features = {}
        
        try:
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Basic color statistics
            mean_color = np.mean(img_array, axis=(0, 1))
            std_color = np.std(img_array, axis=(0, 1))
            
            features['color_red_mean'] = float(mean_color[0]) / 255.0
            features['color_green_mean'] = float(mean_color[1]) / 255.0
            features['color_blue_mean'] = float(mean_color[2]) / 255.0
            features['color_red_std'] = float(std_color[0]) / 255.0
            features['color_green_std'] = float(std_color[1]) / 255.0
            features['color_blue_std'] = float(std_color[2]) / 255.0
            
            # Overall brightness and contrast
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            features['brightness'] = float(np.mean(gray)) / 255.0
            features['contrast'] = float(np.std(gray)) / 255.0
            
            # Color diversity (entropy)
            hist_r = cv2.calcHist([img_array], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([img_array], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([img_array], [2], None, [32], [0, 256])
            
            # Normalize and calculate entropy
            hist_r = hist_r.flatten() / np.sum(hist_r)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_b = hist_b.flatten() / np.sum(hist_b)
            
            features['color_entropy_r'] = -np.sum(hist_r * np.log2(hist_r + 1e-10))
            features['color_entropy_g'] = -np.sum(hist_g * np.log2(hist_g + 1e-10))
            features['color_entropy_b'] = -np.sum(hist_b * np.log2(hist_b + 1e-10))
            
            # Dominant color analysis
            pixels = img_array.reshape(-1, 3)
            unique_colors = len(np.unique(pixels.view(np.dtype((np.void, pixels.dtype.itemsize*pixels.shape[1])))))
            features['unique_colors'] = min(unique_colors / (img_array.shape[0] * img_array.shape[1]), 1.0)
            
            # Background detection (corners are likely background)
            h, w = gray.shape
            corner_size = min(h, w) // 10
            corners = [
                gray[:corner_size, :corner_size],
                gray[:corner_size, -corner_size:],
                gray[-corner_size:, :corner_size],
                gray[-corner_size:, -corner_size:]
            ]
            corner_mean = np.mean([np.mean(corner) for corner in corners])
            features['background_brightness'] = float(corner_mean) / 255.0
            features['is_white_background'] = 1.0 if corner_mean > 240 else 0.0
            
        except Exception:
            # Default values if processing fails
            features = {
                'color_red_mean': 0.5, 'color_green_mean': 0.5, 'color_blue_mean': 0.5,
                'color_red_std': 0.1, 'color_green_std': 0.1, 'color_blue_std': 0.1,
                'brightness': 0.5, 'contrast': 0.1, 'color_entropy_r': 3.0,
                'color_entropy_g': 3.0, 'color_entropy_b': 3.0, 'unique_colors': 0.5,
                'background_brightness': 0.5, 'is_white_background': 0.0
            }
        
        return features

    def extract_quality_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract image quality features"""
        features = {}
        
        try:
            # Image dimensions
            width, height = image.size
            features['actual_width'] = width
            features['actual_height'] = height
            features['actual_area'] = width * height
            features['actual_aspect_ratio'] = width / max(height, 1)
            
            # Image quality metrics
            img_array = np.array(image.convert('L'))  # Grayscale
            
            # Blur detection using Laplacian variance
            blur_metric = cv2.Laplacian(img_array, cv2.CV_64F).var()
            features['blur_metric'] = min(blur_metric / 1000.0, 1.0)  # Normalize
            features['is_blurry'] = 1.0 if blur_metric < 100 else 0.0
            
            # Edge density (indicates detail level)
            edges = cv2.Canny(img_array, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            features['edge_density'] = edge_density
            features['high_detail'] = 1.0 if edge_density > 0.1 else 0.0
            
            # Noise estimation
            noise = cv2.fastNlMeansDenoising(img_array) - img_array
            features['noise_level'] = min(np.std(noise) / 50.0, 1.0)
            
            # Resolution categorization
            total_pixels = width * height
            features['is_thumbnail'] = 1.0 if total_pixels < 10000 else 0.0
            features['is_low_res'] = 1.0 if total_pixels < 100000 else 0.0
            features['is_medium_res'] = 1.0 if 100000 <= total_pixels < 1000000 else 0.0
            features['is_high_res'] = 1.0 if total_pixels >= 1000000 else 0.0
            
        except Exception:
            # Default values
            features = {
                'actual_width': 300, 'actual_height': 300, 'actual_area': 90000,
                'actual_aspect_ratio': 1.0, 'blur_metric': 0.5, 'is_blurry': 0.0,
                'edge_density': 0.05, 'high_detail': 0.0, 'noise_level': 0.1,
                'is_thumbnail': 0.0, 'is_low_res': 0.0, 'is_medium_res': 1.0, 'is_high_res': 0.0
            }
        
        return features

    def extract_composition_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract composition and aesthetic features"""
        features = {}
        
        try:
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            
            # Center vs edge brightness distribution
            center_h, center_w = h//4, w//4
            center_region = gray[center_h:3*center_h, center_w:3*center_w]
            edge_region = np.concatenate([
                gray[:center_h, :].flatten(),
                gray[3*center_h:, :].flatten(),
                gray[:, :center_w].flatten(),
                gray[:, 3*center_w:].flatten()
            ])
            
            features['center_brightness'] = float(np.mean(center_region)) / 255.0
            features['edge_brightness'] = float(np.mean(edge_region)) / 255.0
            features['center_focus'] = features['center_brightness'] - features['edge_brightness']
            
            # Symmetry analysis (horizontal)
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            min_width = min(left_half.shape[1], right_half.shape[1])
            symmetry_diff = np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width]))
            features['horizontal_symmetry'] = 1.0 - min(symmetry_diff / 255.0, 1.0)
            
            # Professional photography indicators
            features['is_centered'] = 1.0 if abs(features['center_focus']) > 0.1 else 0.0
            features['good_lighting'] = 1.0 if 0.3 < features['center_brightness'] < 0.8 else 0.0
            features['studio_quality'] = 1.0 if (features['is_centered'] and features['good_lighting']) else 0.0
            
            # Rule of thirds analysis
            third_h, third_w = h//3, w//3
            thirds_brightness = [
                np.mean(gray[i*third_h:(i+1)*third_h, j*third_w:(j+1)*third_w])
                for i in range(3) for j in range(3)
            ]
            features['thirds_variance'] = float(np.var(thirds_brightness)) / (255.0**2)
            
        except Exception:
            # Default values
            features = {
                'center_brightness': 0.5, 'edge_brightness': 0.5, 'center_focus': 0.0,
                'horizontal_symmetry': 0.5, 'is_centered': 0.0, 'good_lighting': 0.0,
                'studio_quality': 0.0, 'thirds_variance': 0.1
            }
        
        return features

    def extract_comprehensive_image_features(self, image_url: str, sample_id: str) -> Dict[str, float]:
        """Extract all image features"""
        # Download image
        image = self.download_image(image_url, sample_id)
        
        if image is None:
            return self._default_image_features()
        
        # Extract all feature types
        features = {}
        features.update(self.extract_color_features(image))
        features.update(self.extract_quality_features(image))
        features.update(self.extract_composition_features(image))
        
        # Add download success indicator
        features['image_downloaded'] = 1.0
        
        return features

    def _default_image_features(self) -> Dict[str, float]:
        """Default features for failed downloads"""
        return {
            # Color features
            'color_red_mean': 0.5, 'color_green_mean': 0.5, 'color_blue_mean': 0.5,
            'color_red_std': 0.1, 'color_green_std': 0.1, 'color_blue_std': 0.1,
            'brightness': 0.5, 'contrast': 0.1, 'color_entropy_r': 3.0,
            'color_entropy_g': 3.0, 'color_entropy_b': 3.0, 'unique_colors': 0.5,
            'background_brightness': 0.5, 'is_white_background': 0.0,
            
            # Quality features
            'actual_width': 300, 'actual_height': 300, 'actual_area': 90000,
            'actual_aspect_ratio': 1.0, 'blur_metric': 0.5, 'is_blurry': 0.0,
            'edge_density': 0.05, 'high_detail': 0.0, 'noise_level': 0.1,
            'is_thumbnail': 0.0, 'is_low_res': 0.0, 'is_medium_res': 1.0, 'is_high_res': 0.0,
            
            # Composition features
            'center_brightness': 0.5, 'edge_brightness': 0.5, 'center_focus': 0.0,
            'horizontal_symmetry': 0.5, 'is_centered': 0.0, 'good_lighting': 0.0,
            'studio_quality': 0.0, 'thirds_variance': 0.1,
            
            # Status
            'image_downloaded': 0.0
        }

class EnhancedMultiModalExtractor:
    """Enhanced feature extraction with computer vision"""
    
    def __init__(self, token_price_map: Dict[str, float] | None = None):
        self._keyword_groups: Dict[str, List[str]] = {
            'qual_budget': ['budget', 'affordable', 'economy', 'basic', 'value', 'cheap', 'discount', 'clearance'],
            'qual_premium': ['premium', 'luxury', 'deluxe', 'professional', 'authentic', 'genuine', 'elite', 'superior'],
            'cat_automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'oil', 'brake', 'automotive'],
            'cat_jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'gold', 'silver', 'diamond', 'platinum'],
            'cat_electronics': ['electronic', 'device', 'smart', 'digital', 'tech', 'wireless', 'bluetooth'],
            'cat_health': ['health', 'vitamin', 'supplement', 'organic', 'wellness', 'medical'],
            'cat_tools': ['tool', 'drill', 'hammer', 'equipment', 'hardware', 'workshop'],
            'cat_home': ['home', 'kitchen', 'furniture', 'house', 'decor', 'living'],
            'cat_food': ['food', 'snack', 'gourmet', 'cooking', 'recipe', 'ingredient'],
            'brand_indicators': ['brand', 'official', 'licensed', 'certified', 'authorized']
        }
        self._token_price_map = token_price_map or {}
        self.cv_extractor = ComputerVisionExtractor()

    def extract_enhanced_features(self, catalog_content: str, image_link: str, sample_id: str) -> Dict[str, float]:
        """Extract comprehensive features including computer vision"""
        text = str(catalog_content or '')
        tl = text.lower()
        features = {}

        # Text features
        features.update(self._extract_text_features(text, tl))
        
        # Computer vision features from actual images
        features.update(self.cv_extractor.extract_comprehensive_image_features(image_link, sample_id))
        
        # Cross-modal features
        features.update(self._extract_cross_modal_features(text, image_link))
        
        return features

    def _extract_text_features(self, text: str, tl: str) -> Dict[str, float]:
        """Extract text features (condensed version)"""
        features = {}
        
        # Basic statistics
        features['text_length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['unique_word_ratio'] = len(set(words)) / max(len(words), 1)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Enhanced number extraction
        numbers = self._extract_numbers(text)
        features['number_count'] = len(numbers)
        features['max_number'] = max(numbers) if numbers else 0
        features['number_std'] = float(np.std(numbers)) if len(numbers) > 1 else 0
        
        # Quantity and pricing indicators
        features['quantity'] = self._extract_quantity(tl)
        features['has_dollar_sign'] = 1 if '$' in text else 0
        features['has_price_words'] = 1 if any(w in tl for w in ['price', 'cost', 'value', 'msrp']) else 0
        
        # Category features
        for cat, keywords in self._keyword_groups.items():
            features[cat] = sum(1 for kw in keywords if kw in tl)
        
        # Token priors
        tokens = re.findall(r"[a-zA-Z0-9$%\.\-]+", tl)
        priors = [self._token_price_map.get(t, 0) for t in tokens if t in self._token_price_map]
        features['token_prior_mean'] = float(np.mean(priors)) if priors else 0
        features['token_prior_max'] = float(np.max(priors)) if priors else 0
        features['token_prior_count'] = len(priors)
        
        return features

    def _extract_cross_modal_features(self, text: str, image_link: str) -> Dict[str, float]:
        """Extract cross-modal consistency features"""
        features = {}
        
        if not text or not image_link:
            return {'cross_modal_consistency': 0.0}
        
        text_lower = text.lower()
        url_lower = image_link.lower()
        
        # Word overlap
        text_words = set(re.findall(r'[a-zA-Z]+', text_lower))
        url_words = set(re.findall(r'[a-zA-Z]+', url_lower))
        
        if text_words and url_words:
            intersection = text_words.intersection(url_words)
            features['cross_modal_word_overlap'] = len(intersection) / min(len(text_words), len(url_words))
        else:
            features['cross_modal_word_overlap'] = 0.0
        
        return features

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        numbers = []
        for match in re.findall(r'\d+\.?\d*', text):
            try:
                val = float(match)
                if 0.01 <= val <= 100000:
                    numbers.append(val)
            except ValueError:
                continue
        return sorted(set(numbers))

    def _extract_quantity(self, text_lower: str) -> float:
        """Extract quantity from text"""
        patterns = [
            r'pack of (\d+)', r'(\d+)\s*pack', r'(\d+)\s*count',
            r'(\d+)\s*piece', r'quantity:?\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    return float(np.clip(float(match.group(1)), 1, 100))
                except (ValueError, IndexError):
                    continue
        return 1.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe to feature matrix with progress tracking"""
        features_list = []
        total_rows = len(df)
        
        print(f"🔍 Processing {total_rows} samples with computer vision...")
        
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx % 1000 == 0:
                print(f"  Progress: {idx}/{total_rows} ({idx/total_rows*100:.1f}%)")
            
            features = self.extract_enhanced_features(
                row.get('catalog_content', ''),
                row.get('image_link', ''),
                str(row.get('sample_id', f'sample_{idx}'))
            )
            features_list.append(features)
        
        print(f"✅ Completed processing {total_rows} samples")
        return pd.DataFrame(features_list).fillna(0)

def train_computer_vision_model(train_csv: str, sample_size: int = 30000, val_size: int = 5000):
    """Train model with computer vision features"""
    start_time = time.time()
    print(f"🖼️ Loading {sample_size + val_size} samples for computer vision analysis...")
    df = pd.read_csv(train_csv, nrows=sample_size + val_size)
    
    train_df = df.iloc[:sample_size].copy()
    val_df = df.iloc[sample_size:sample_size+val_size].copy()
    
    # Build token priors
    print("Building token priors...")
    token_price_map = {}
    for _, r in train_df.iterrows():
        text = str(r.get('catalog_content', '') or '').lower()
        price = float(r['price'])
        for tok in set(re.findall(r"[a-zA-Z0-9$%\.\-]+", text)):
            if 0.1 <= price <= 3000 and len(tok) >= 2:
                token_price_map.setdefault(tok, []).append(price)
    
    token_price_map = {k: float(np.median(v)) for k, v in token_price_map.items() if len(v) >= 3}
    print(f"Built price priors for {len(token_price_map)} tokens")
    
    # Extract computer vision features
    print("🔍 Extracting computer vision features...")
    fx = EnhancedMultiModalExtractor(token_price_map=token_price_map)
    X_features_train = fx.transform(train_df)
    X_features_val = fx.transform(val_df)
    
    print(f"Extracted {X_features_train.shape[1]} features (including computer vision)")
    
    # Check download success rate
    download_rate = X_features_train['image_downloaded'].mean()
    print(f"📊 Image download success rate: {download_rate*100:.1f}%")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_features_train)
    X_scaled_val = scaler.transform(X_features_val)
    
    # Prepare targets
    y_train = np.log1p(train_df['price'].values)
    y_val_true = val_df['price'].values
    
    # Dynamic clipping
    q_low, q_high = np.percentile(train_df['price'].values, [0.05, 99.95])
    
    # Text processing
    print("Text vectorization...")
    vectorizer = TfidfVectorizer(
        analyzer='word', lowercase=True, ngram_range=(1, 3),
        min_df=2, max_df=0.9, max_features=200000, strip_accents='unicode'
    )
    
    X_txt_train = vectorizer.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val = vectorizer.transform(val_df['catalog_content'].fillna('').astype(str))
    
    # Train ensemble
    print("Training computer vision enhanced ensemble...")
    models = {}
    
    # Text models
    models['ridge'] = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    models['ridge'].fit(X_txt_train, y_train)
    
    models['elastic'] = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000)
    models['elastic'].fit(X_txt_train, y_train)
    
    # Feature models with computer vision
    models['histgb'] = HistGradientBoostingRegressor(
        loss='absolute_error', max_depth=8, max_iter=800,
        learning_rate=0.03, random_state=RANDOM_STATE
    )
    models['histgb'].fit(X_scaled_train, y_train)
    
    models['rf'] = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    models['rf'].fit(X_scaled_train, y_train)
    
    models['extratrees'] = ExtraTreesRegressor(
        n_estimators=200, max_depth=12, min_samples_split=4,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    models['extratrees'].fit(X_scaled_train, y_train)
    
    # Neural network
    models['mlp'] = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128), activation='relu',
        solver='adam', alpha=0.001, max_iter=300, random_state=RANDOM_STATE
    )
    models['mlp'].fit(X_scaled_train, y_train)
    
    # Get predictions
    print("Optimizing ensemble with computer vision features...")
    predictions = {}
    predictions['ridge'] = models['ridge'].predict(X_txt_val)
    predictions['elastic'] = models['elastic'].predict(X_txt_val)
    predictions['histgb'] = models['histgb'].predict(X_scaled_val)
    predictions['rf'] = models['rf'].predict(X_scaled_val)
    predictions['extratrees'] = models['extratrees'].predict(X_scaled_val)
    predictions['mlp'] = models['mlp'].predict(X_scaled_val)
    
    # Optimize weights
    best_weights = None
    best_smape = float('inf')
    
    model_names = list(predictions.keys())
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    search_count = 0
    for w1 in weight_options[::2]:
        for w2 in weight_options[::2]:
            for w3 in weight_options[::2]:
                for w4 in weight_options[::2]:
                    for w5 in weight_options[::2]:
                        w6 = 1.0 - (w1 + w2 + w3 + w4 + w5)
                        if w6 < 0 or w6 > 1:
                            continue
                        
                        weights = [w1, w2, w3, w4, w5, w6]
                        
                        pred_log_blend = sum(w * predictions[model] for w, model in zip(weights, model_names))
                        pred_price_try = np.expm1(pred_log_blend)
                        pred_price_try = np.clip(pred_price_try, q_low * 0.6, q_high * 1.3)
                        
                        s_try = smape(y_val_true, pred_price_try)
                        search_count += 1
                        
                        if s_try < best_smape:
                            best_smape = s_try
                            best_weights = weights
                            print(f"  New best CV-SMAPE: {s_try:.3f}% (search #{search_count})")
    
    s = best_smape
    corr = np.corrcoef(y_val_true, np.expm1(sum(w * predictions[model] for w, model in zip(best_weights, model_names))))[0, 1]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=== Computer Vision Enhanced Results ===")
    weight_str = ", ".join([f"{name}:{w:.2f}" for name, w in zip(model_names, best_weights)])
    print(f"Optimal weights: {weight_str}")
    print(f"SMAPE: {s:.2f}%")
    print(f"Correlation: {corr:.4f}")
    print(f"Image download success: {download_rate*100:.1f}%")
    print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    artifacts = {
        'fx': fx, 'scaler': scaler, 'vectorizer': vectorizer,
        'models': models, 'model_names': model_names, 'weights': best_weights,
        'clip_low': float(q_low), 'clip_high': float(q_high)
    }
    
    return s, artifacts

def predict_cv_test(test_csv: str, artifacts: dict, out_path: str):
    """Generate test predictions using computer vision model"""
    print("🖼️ Generating computer vision test predictions...")
    test_df = pd.read_csv(test_csv)
    
    # Extract features
    fx = artifacts['fx']
    X_features_test = fx.transform(test_df)
    X_scaled_test = artifacts['scaler'].transform(X_features_test)
    
    # Text processing
    X_txt_test = artifacts['vectorizer'].transform(test_df['catalog_content'].fillna('').astype(str))
    
    # Get predictions
    models = artifacts['models']
    model_names = artifacts['model_names']
    predictions = {}
    
    predictions['ridge'] = models['ridge'].predict(X_txt_test)
    predictions['elastic'] = models['elastic'].predict(X_txt_test)
    predictions['histgb'] = models['histgb'].predict(X_scaled_test)
    predictions['rf'] = models['rf'].predict(X_scaled_test)
    predictions['extratrees'] = models['extratrees'].predict(X_scaled_test)
    predictions['mlp'] = models['mlp'].predict(X_scaled_test)
    
    # Weighted ensemble
    weights = artifacts['weights']
    pred_log_blend = sum(w * predictions[model] for w, model in zip(weights, model_names))
    pred_price = np.expm1(pred_log_blend)
    pred_price = np.clip(pred_price, artifacts['clip_low'], artifacts['clip_high'])
    
    # Save predictions
    out = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': np.round(pred_price, 2)})
    out.to_csv(out_path, index=False)
    print(f"Computer vision predictions saved to {out_path}")

if __name__ == "__main__":
    print("🖼️ Computer Vision Enhanced Model")
    print("Downloading and analyzing actual images from dataset URLs")
    print("Target: Push SMAPE from 50.12% to <44%")
    print("Safe within academic rules - using only provided dataset")
    print("=" * 70)
    
    smape_val, art = train_computer_vision_model('dataset/train.csv', sample_size=30000, val_size=5000)
    
    if smape_val < 55.0:
        predict_cv_test('dataset/test.csv', art, 'dataset/test_out_computer_vision.csv')
    
    print(f"\nFinal Computer Vision SMAPE: {smape_val:.2f}%")
    if smape_val < 44.0:
        print("🎉🎉🎉 TARGET ACHIEVED! <44% SMAPE 🎉🎉🎉")
        print("🖼️ COMPUTER VISION SUCCESS! 🖼️")
    else:
        gap = smape_val - 44.0
        print(f"Gap to target: {gap:.2f} percentage points")
        
        if smape_val < 47.0:
            print("🚀 Excellent improvement with computer vision!")
        elif smape_val < 49.0:
            print("📈 Good improvement! Visual features are helping.")
        else:
            print("🔍 Computer vision providing modest gains.")