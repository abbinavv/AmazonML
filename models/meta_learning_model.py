"""
Meta-Learning Stacking Model: Advanced Second-Level Learning
Target: Push SMAPE from 50.45% to <44% using stacked meta-learning
Uses ONLY existing datasets - no external data integration
Leverages comprehensive image URL features + text analysis
"""

import re
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple
from urllib.parse import urlparse

# Core ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
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
    """Enhanced feature extraction with comprehensive image URL analysis"""
    
    def __init__(self, token_price_map: Dict[str, float] | None = None):
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
        self._token_price_map = token_price_map or {}

    def extract_enhanced_features(self, catalog_content: str, image_link: str) -> Dict[str, float]:
        """Extract comprehensive features with advanced image URL analysis"""
        text = str(catalog_content or '')
        tl = text.lower()
        features = {}

        # Enhanced text features
        features.update(self._extract_text_features(text, tl))
        
        # Comprehensive image URL features  
        features.update(self._extract_comprehensive_image_features(image_link))
        
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
        
        # Advanced text patterns
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['title_case_words'] = sum(1 for w in words if w.istitle()) / max(len(words), 1)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['parentheses_count'] = text.count('(') + text.count(')')
        
        # Enhanced number extraction
        numbers = self._extract_enhanced_numbers(text)
        features['number_count'] = len(numbers)
        features['max_number'] = max(numbers) if numbers else 0
        features['min_number'] = min(numbers) if numbers else 0
        features['number_range'] = features['max_number'] - features['min_number']
        features['number_std'] = float(np.std(numbers)) if len(numbers) > 1 else 0
        features['number_variance'] = float(np.var(numbers)) if len(numbers) > 1 else 0
        
        # Currency and price indicators
        features['has_dollar_sign'] = 1 if '$' in text else 0
        features['has_percent_sign'] = 1 if '%' in text else 0
        features['has_price_words'] = 1 if any(w in tl for w in ['price', 'cost', 'value', 'msrp', 'retail']) else 0
        features['has_discount_words'] = 1 if any(w in tl for w in ['sale', 'discount', 'off', 'deal', 'save']) else 0
        
        # Enhanced quantity extraction
        qty = self._extract_enhanced_quantity(tl)
        features['quantity'] = qty
        features['quantity_log'] = np.log1p(qty)
        features['is_bulk'] = 1 if qty > 1 else 0
        features['is_large_bulk'] = 1 if qty > 10 else 0
        features['is_massive_bulk'] = 1 if qty > 50 else 0
        
        # Enhanced unit extraction
        units = self._extract_enhanced_units(tl)
        for unit, val in units.items():
            features[f'unit_{unit}'] = val
        
        # Category features with enhanced scoring
        category_scores = {}
        for cat, keywords in self._keyword_groups.items():
            score = sum(1 for kw in keywords if kw in tl)
            features[cat] = score
            # Weighted scoring for longer matches
            weighted_score = sum(len(kw) for kw in keywords if kw in tl)
            features[f'{cat}_weighted'] = weighted_score
            if cat.startswith('cat_'):
                category_scores[cat] = score
        
        # Dominant category analysis
        if category_scores:
            dominant_cat = max(category_scores, key=category_scores.get)
            for cat in category_scores:
                features[f'{cat}_dominant'] = 1 if cat == dominant_cat else 0
            features['category_diversity'] = len([c for c in category_scores.values() if c > 0])
        
        # Enhanced token-level priors
        tokens = re.findall(r"[a-zA-Z0-9$%\.\-]+", tl)
        priors = [self._token_price_map.get(t, 0) for t in tokens if t in self._token_price_map]
        features['token_prior_mean'] = float(np.mean(priors)) if priors else 0
        features['token_prior_median'] = float(np.median(priors)) if priors else 0
        features['token_prior_max'] = float(np.max(priors)) if priors else 0
        features['token_prior_min'] = float(np.min(priors)) if priors else 0
        features['token_prior_std'] = float(np.std(priors)) if len(priors) > 1 else 0
        features['token_prior_range'] = features['token_prior_max'] - features['token_prior_min']
        features['token_prior_count'] = len(priors)
        features['token_coverage'] = len(priors) / max(len(tokens), 1)
        
        # Price distribution features
        if priors:
            q25, q75 = np.percentile(priors, [25, 75])
            features['token_prior_q25'] = float(q25)
            features['token_prior_q75'] = float(q75)
            features['token_prior_iqr'] = float(q75 - q25)
        else:
            features['token_prior_q25'] = 0
            features['token_prior_q75'] = 0
            features['token_prior_iqr'] = 0
        
        return features

    def _extract_comprehensive_image_features(self, image_link: str) -> Dict[str, float]:
        """Comprehensive image URL feature extraction"""
        features = {}
        
        if pd.isna(image_link) or not isinstance(image_link, str) or len(image_link) < 10:
            return self._default_image_features()
        
        try:
            url_lower = image_link.lower()
            parsed = urlparse(image_link)
            
            # Enhanced URL structure analysis
            features['img_url_length'] = min(len(image_link), 2000)
            features['img_domain_length'] = len(parsed.netloc)
            features['img_path_depth'] = len([x for x in parsed.path.split('/') if x])
            features['img_query_params'] = len(parsed.query.split('&')) if parsed.query else 0
            features['img_fragment_present'] = 1 if parsed.fragment else 0
            features['img_is_https'] = 1 if image_link.startswith('https') else 0
            features['img_port_specified'] = 1 if parsed.port else 0
            
            # Advanced platform detection with confidence scoring
            platform_patterns = {
                'amazon': ['amazon', 'amzn', 'a.co', 'aws', 'amazonas'],
                'shopify': ['shopify', 'cdn.shopify', 'shopifycdn'],
                'cloudinary': ['cloudinary', 'res.cloudinary', 'cloudinary.com'],
                'imgur': ['imgur', 'i.imgur', 'imgur.com'],
                'cdn': ['cdn', 'static', 'assets', 'media', 'images', 'content'],
                'google': ['google', 'gstatic', 'googleusercontent'],
                'facebook': ['facebook', 'fbcdn', 'scontent'],
                'wordpress': ['wordpress', 'wp-content', 'gravatar']
            }
            
            platform_confidence = {}
            for platform, keywords in platform_patterns.items():
                confidence = sum(2 if kw in parsed.netloc else 1 for kw in keywords if kw in url_lower)
                features[f'img_platform_{platform}'] = confidence
                platform_confidence[platform] = confidence
            
            # Dominant platform
            if platform_confidence:
                dominant = max(platform_confidence, key=platform_confidence.get)
                for platform in platform_confidence:
                    features[f'img_{platform}_dominant'] = 1 if platform == dominant else 0
            
            # Enhanced format detection and quality assessment
            formats = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'svg', 'bmp', 'tiff', 'avif']
            format_quality = {
                'avif': 0.95, 'svg': 0.90, 'webp': 0.85, 'png': 0.80, 
                'tiff': 0.75, 'jpeg': 0.70, 'jpg': 0.70, 'gif': 0.40, 'bmp': 0.30
            }
            
            detected_formats = []
            for fmt in formats:
                if f'.{fmt}' in url_lower:
                    features[f'img_format_{fmt}'] = 1
                    detected_formats.append(fmt)
                else:
                    features[f'img_format_{fmt}'] = 0
            
            features['img_format_quality'] = max([format_quality.get(fmt, 0.5) for fmt in detected_formats], default=0.5)
            features['img_multiple_formats'] = 1 if len(detected_formats) > 1 else 0
            features['img_format_count'] = len(detected_formats)
            
            # Advanced dimension extraction and analysis
            dimensions = self._extract_enhanced_dimensions(image_link)
            features['img_width'] = dimensions.get('width', 0)
            features['img_height'] = dimensions.get('height', 0)
            features['img_area'] = features['img_width'] * features['img_height']
            features['img_perimeter'] = 2 * (features['img_width'] + features['img_height'])
            features['img_aspect_ratio'] = features['img_width'] / max(features['img_height'], 1)
            features['img_diagonal'] = np.sqrt(features['img_width']**2 + features['img_height']**2)
            
            # Dimension categorization
            features['img_is_square'] = 1 if abs(features['img_aspect_ratio'] - 1.0) < 0.1 else 0
            features['img_is_portrait'] = 1 if features['img_aspect_ratio'] < 0.8 else 0
            features['img_is_landscape'] = 1 if features['img_aspect_ratio'] > 1.2 else 0
            features['img_is_ultrawide'] = 1 if features['img_aspect_ratio'] > 2.0 else 0
            features['img_is_tall'] = 1 if features['img_aspect_ratio'] < 0.5 else 0
            
            # Advanced size categorization
            max_dim = max(features['img_width'], features['img_height'])
            features['img_size_thumbnail'] = 1 if 0 < max_dim <= 100 else 0
            features['img_size_small'] = 1 if 100 < max_dim <= 300 else 0
            features['img_size_medium'] = 1 if 300 < max_dim <= 800 else 0
            features['img_size_large'] = 1 if 800 < max_dim <= 1500 else 0
            features['img_size_xlarge'] = 1 if 1500 < max_dim <= 3000 else 0
            features['img_size_huge'] = 1 if max_dim > 3000 else 0
            
            # Resolution quality indicators
            total_pixels = features['img_area']
            features['img_resolution_low'] = 1 if total_pixels < 100000 else 0  # < 0.1MP
            features['img_resolution_medium'] = 1 if 100000 <= total_pixels < 1000000 else 0  # 0.1-1MP
            features['img_resolution_high'] = 1 if 1000000 <= total_pixels < 5000000 else 0  # 1-5MP
            features['img_resolution_ultra'] = 1 if total_pixels >= 5000000 else 0  # 5MP+
            
            # Quality and professional indicators
            quality_terms = ['hd', 'hq', 'high', 'quality', 'resolution', 'dpi', 'retina', 'professional', '4k', '8k']
            features['img_quality_score'] = sum(2 if term in parsed.path else 1 for term in quality_terms if term in url_lower)
            
            professional_terms = ['studio', 'professional', 'product', 'catalog', 'gallery', 'portrait', 'commercial']
            features['img_professional_score'] = sum(1 for term in professional_terms if term in url_lower)
            
            # Enhanced filename analysis
            filename = parsed.path.split('/')[-1] if parsed.path else ''
            features['img_filename_length'] = len(filename)
            features['img_has_hash'] = 1 if len(re.findall(r'[a-f0-9]{8,}', filename.lower())) > 0 else 0
            features['img_has_timestamp'] = 1 if re.search(r'\d{8,}', filename) else 0
            features['img_has_uuid'] = 1 if re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', filename.lower()) else 0
            features['img_filename_numbers'] = len(re.findall(r'\d+', filename))
            features['img_filename_words'] = len(re.findall(r'[a-zA-Z]+', filename))
            
            # URL path analysis
            path_segments = [seg for seg in parsed.path.split('/') if seg]
            features['img_path_segments'] = len(path_segments)
            features['img_has_product_path'] = 1 if any(seg in ['product', 'item', 'p', 'catalog'] for seg in path_segments) else 0
            features['img_has_image_path'] = 1 if any(seg in ['images', 'img', 'photos', 'pics'] for seg in path_segments) else 0
            
            # Security and optimization indicators
            features['img_has_webp_support'] = 1 if 'webp' in url_lower else 0
            features['img_has_optimization'] = 1 if any(term in url_lower for term in ['opt', 'compressed', 'resize', 'thumb']) else 0
            features['img_has_version'] = 1 if re.search(r'v\d+', url_lower) else 0
            
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
        
        # Enhanced word overlap analysis
        text_words = set(re.findall(r'[a-zA-Z]+', text_lower))
        url_words = set(re.findall(r'[a-zA-Z]+', url_lower))
        
        if text_words and url_words:
            intersection = text_words.intersection(url_words)
            union = text_words.union(url_words)
            features['cross_modal_word_overlap'] = len(intersection) / min(len(text_words), len(url_words))
            features['cross_modal_jaccard'] = len(intersection) / len(union)
            features['cross_modal_shared_words'] = len(intersection)
        else:
            features['cross_modal_word_overlap'] = 0.0
            features['cross_modal_jaccard'] = 0.0
            features['cross_modal_shared_words'] = 0
        
        # Brand consistency analysis
        brand_words = ['brand', 'official', 'authentic', 'genuine', 'licensed']
        text_has_brand = any(word in text_lower for word in brand_words)
        url_has_brand_indicator = any(term in url_lower for term in brand_words)
        features['cross_modal_brand_consistency'] = 1 if text_has_brand == url_has_brand_indicator else 0
        
        # Quality consistency
        quality_words = ['premium', 'luxury', 'professional', 'high', 'quality']
        text_quality_score = sum(1 for word in quality_words if word in text_lower)
        url_quality_score = sum(1 for word in quality_words if word in url_lower)
        features['cross_modal_quality_consistency'] = 1 if (text_quality_score > 0) == (url_quality_score > 0) else 0
        
        return features

    def _extract_interaction_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create advanced interaction features"""
        interactions = {}
        
        # Text-quantity interactions
        if 'text_length' in features and 'quantity' in features:
            interactions['text_qty_interaction'] = features['text_length'] * np.log1p(features['quantity'])
            interactions['text_qty_ratio'] = features['text_length'] / max(features['quantity'], 1)
        
        # Price-quality interactions
        if 'qual_premium' in features and 'token_prior_mean' in features:
            interactions['premium_price_interaction'] = features['qual_premium'] * features['token_prior_mean']
            interactions['premium_price_ratio'] = features['qual_premium'] / max(features['token_prior_mean'], 0.1)
        
        # Category-size interactions
        for cat in ['cat_electronics', 'cat_jewelry', 'cat_automotive', 'cat_home']:
            if cat in features and 'img_area' in features:
                interactions[f'{cat}_size_interaction'] = features[cat] * np.log1p(features['img_area'])
        
        # Brand-platform interactions
        if 'brand_indicators' in features and 'img_platform_amazon' in features:
            interactions['brand_platform_interaction'] = features['brand_indicators'] * features['img_platform_amazon']
        
        # Image quality-text quality interactions
        if 'img_quality_score' in features and 'qual_premium' in features:
            interactions['img_text_quality_interaction'] = features['img_quality_score'] * features['qual_premium']
        
        # Advanced ratio features
        if 'word_count' in features and 'img_area' in features:
            interactions['text_image_ratio'] = features['word_count'] / max(np.log1p(features['img_area']), 1)
        
        return interactions

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
        """Enhanced quantity extraction with more patterns"""
        patterns = [
            r'pack of (\d+)', r'(\d+)\s*pack', r'(\d+)\s*count', r'set of (\d+)',
            r'(\d+)\s*piece', r'(\d+)\s*items?', r'(\d+)\s*units?',
            r'quantity:?\s*(\d+)', r'qty:?\s*(\d+)', r'(\d+)\s*ct',
            r'box of (\d+)', r'case of (\d+)', r'bundle of (\d+)',
            r'(\d+)\s*dozen', r'(\d+)\s*pair'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    qty = float(match.group(1))
                    if 'dozen' in pattern:
                        qty *= 12
                    elif 'pair' in pattern:
                        qty *= 2
                    return float(np.clip(qty, 1, 1000))
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
            r'(\d+)_(\d+)_',                   # 800_600_
            r'img_(\d+)x(\d+)',               # img_800x600
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, url, re.IGNORECASE)
            for match in matches:
                try:
                    w, h = int(match[0]), int(match[1])
                    if 10 <= w <= 20000 and 10 <= h <= 20000:
                        dimensions['width'] = max(dimensions['width'], w)
                        dimensions['height'] = max(dimensions['height'], h)
                except (ValueError, IndexError):
                    continue
        
        return dimensions

    def _default_image_features(self) -> Dict[str, float]:
        """Default image features for missing/invalid URLs"""
        return {
            'img_url_length': 0, 'img_domain_length': 0, 'img_path_depth': 0, 'img_query_params': 0,
            'img_is_https': 0, 'img_width': 300, 'img_height': 300, 'img_area': 90000, 'img_aspect_ratio': 1.0,
            'img_is_square': 1, 'img_is_portrait': 0, 'img_is_landscape': 0,
            'img_size_small': 0, 'img_size_medium': 1, 'img_size_large': 0,
            'img_quality_score': 0, 'img_professional_score': 0, 'img_format_quality': 0.5,
            'img_filename_length': 0, 'img_has_hash': 0, 'img_has_timestamp': 0,
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

def get_out_of_fold_predictions(X, y, models_config: List[Tuple], n_folds: int = 5) -> np.ndarray:
    """Generate out-of-fold predictions for meta-learning"""
    n_samples = X.shape[0]
    n_models = len(models_config)
    oof_predictions = np.zeros((n_samples, n_models))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    for model_idx, (model_name, model_factory, data_type) in enumerate(models_config):
        print(f"  Generating OOF predictions for {model_name}...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            if data_type == 'dense':
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            else:  # sparse
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            
            y_train_fold = y[train_idx]
            
            # Create and train model
            model = model_factory()
            model.fit(X_train_fold, y_train_fold)
            
            # Predict on validation fold
            oof_predictions[val_idx, model_idx] = model.predict(X_val_fold)
    
    return oof_predictions

def train_meta_learning_model(train_csv: str, sample_size: int = 45000, val_size: int = 5000):
    """Train the meta-learning stacking model"""
    start_time = time.time()
    print(f"Loading {sample_size + val_size} samples for meta-learning...")
    df = pd.read_csv(train_csv, nrows=sample_size + val_size)
    
    train_df = df.iloc[:sample_size].copy()
    val_df = df.iloc[sample_size:sample_size+val_size].copy()
    
    # Build enhanced token priors
    print("Building comprehensive token priors...")
    token_price_map = {}
    for _, r in train_df.iterrows():
        text = str(r.get('catalog_content', '') or '').lower()
        price = float(r['price'])
        for tok in set(re.findall(r"[a-zA-Z0-9$%\.\-]+", text)):
            if 0.1 <= price <= 3000 and len(tok) >= 2:
                token_price_map.setdefault(tok, []).append(price)
    
    # Use median for stability, require fewer samples for broader coverage
    token_price_map = {k: float(np.median(v)) for k, v in token_price_map.items() if len(v) >= 3}
    print(f"Built price priors for {len(token_price_map)} tokens")
    
    # Extract comprehensive features
    print("Extracting comprehensive multi-modal features...")
    fx = AdvancedMultiModalExtractor(token_price_map=token_price_map)
    X_features_train = fx.transform(train_df)
    X_features_val = fx.transform(val_df)
    
    print(f"Extracted {X_features_train.shape[1]} base features")
    
    # Create polynomial interactions for key features
    print("Creating strategic polynomial interactions...")
    important_cols = [
        'text_length', 'word_count', 'quantity', 'token_prior_mean', 'token_prior_max',
        'img_area', 'img_quality_score', 'qual_premium', 'cat_electronics', 'cat_jewelry',
        'cross_modal_word_overlap', 'brand_indicators'
    ]
    important_cols = [col for col in important_cols if col in X_features_train.columns]
    
    if important_cols:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly_train = poly.fit_transform(X_features_train[important_cols])
        X_poly_val = poly.transform(X_features_val[important_cols])
        
        X_combined_train = np.hstack([X_features_train.values, X_poly_train])
        X_combined_val = np.hstack([X_features_val.values, X_poly_val])
    else:
        X_combined_train = X_features_train.values
        X_combined_val = X_features_val.values
        poly = None
    
    print(f"Total features with interactions: {X_combined_train.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_combined_train)
    X_scaled_val = scaler.transform(X_combined_val)
    
    # Prepare targets
    y_train = np.log1p(train_df['price'].values)
    y_val_true = val_df['price'].values
    
    # Dynamic clipping
    q_low, q_high = np.percentile(train_df['price'].values, [0.05, 99.95])
    
    # Enhanced text processing
    print("Enhanced text vectorization...")
    vectorizer_word = TfidfVectorizer(
        analyzer='word', lowercase=True, ngram_range=(1, 3),
        min_df=2, max_df=0.9, max_features=350000, strip_accents='unicode',
        sublinear_tf=True
    )
    vectorizer_char = TfidfVectorizer(
        analyzer='char', ngram_range=(2, 6),
        min_df=2, max_df=0.95, max_features=150000,
        sublinear_tf=True
    )
    
    X_txt_train_word = vectorizer_word.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_word = vectorizer_word.transform(val_df['catalog_content'].fillna('').astype(str))
    X_txt_train_char = vectorizer_char.fit_transform(train_df['catalog_content'].fillna('').astype(str))
    X_txt_val_char = vectorizer_char.transform(val_df['catalog_content'].fillna('').astype(str))
    
    X_txt_train = hstack([X_txt_train_word, X_txt_train_char])
    X_txt_val = hstack([X_txt_val_word, X_txt_val_char])
    
    # SVD for dimensionality reduction
    print("Applying SVD for dense text features...")
    svd = TruncatedSVD(n_components=400, random_state=RANDOM_STATE)
    X_txt_train_svd = svd.fit_transform(X_txt_train)
    X_txt_val_svd = svd.transform(X_txt_val)
    
    # Create comprehensive dense features
    X_dense_train = np.hstack([X_scaled_train, X_txt_train_svd])
    X_dense_val = np.hstack([X_scaled_val, X_txt_val_svd])
    
    print(f"Dense feature matrix: {X_dense_train.shape}")
    
    # Define base models for meta-learning
    print("Generating out-of-fold predictions for meta-learning...")
    
    base_models_config = [
        ('ridge_text', lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE), 'sparse'),
        ('elastic_text', lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=2000), 'sparse'),
        ('histgb_dense', lambda: HistGradientBoostingRegressor(
            loss='absolute_error', max_depth=8, max_iter=800,
            learning_rate=0.03, l2_regularization=0.02, random_state=RANDOM_STATE
        ), 'dense'),
        ('extratrees_dense', lambda: ExtraTreesRegressor(
            n_estimators=250, max_depth=12, min_samples_split=4,
            random_state=RANDOM_STATE, n_jobs=-1
        ), 'dense'),
        ('rf_dense', lambda: RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ), 'dense'),
        ('mlp_dense', lambda: MLPRegressor(
            hidden_layer_sizes=(512, 256, 128), activation='relu',
            solver='adam', alpha=0.001, learning_rate='adaptive',
            max_iter=300, random_state=RANDOM_STATE
        ), 'dense'),
        ('sgd_text', lambda: SGDRegressor(
            loss='huber', alpha=1e-5, max_iter=2000, random_state=RANDOM_STATE
        ), 'sparse')
    ]
    
    # Generate out-of-fold predictions
    oof_text_models = [(name, factory, 'sparse') for name, factory, dtype in base_models_config if dtype == 'sparse']
    oof_dense_models = [(name, factory, 'dense') for name, factory, dtype in base_models_config if dtype == 'dense']
    
    # Get OOF predictions for text models
    oof_text_preds = get_out_of_fold_predictions(X_txt_train, y_train, oof_text_models, n_folds=5)
    
    # Get OOF predictions for dense models
    oof_dense_preds = get_out_of_fold_predictions(X_dense_train, y_train, oof_dense_models, n_folds=5)
    
    # Combine all OOF predictions
    oof_predictions = np.hstack([oof_text_preds, oof_dense_preds])
    
    print(f"Meta-learning feature matrix: {oof_predictions.shape}")
    
    # Train meta-models on OOF predictions
    print("Training meta-models...")
    
    meta_models = {}
    
    # Meta-model 1: Ridge regression
    meta_models['meta_ridge'] = Ridge(alpha=0.1, random_state=RANDOM_STATE)
    meta_models['meta_ridge'].fit(oof_predictions, y_train)
    
    # Meta-model 2: Gradient boosting
    meta_models['meta_histgb'] = HistGradientBoostingRegressor(
        loss='absolute_error', max_depth=6, max_iter=500,
        learning_rate=0.05, l2_regularization=0.01, random_state=RANDOM_STATE
    )
    meta_models['meta_histgb'].fit(oof_predictions, y_train)
    
    # Meta-model 3: Extra trees
    meta_models['meta_extratrees'] = ExtraTreesRegressor(
        n_estimators=150, max_depth=8, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    meta_models['meta_extratrees'].fit(oof_predictions, y_train)
    
    # Train base models on full training data for validation predictions
    print("Training base models on full data...")
    full_models = {}
    
    # Train text models
    for name, factory, dtype in oof_text_models:
        model = factory()
        model.fit(X_txt_train, y_train)
        full_models[name] = (model, 'sparse')
    
    # Train dense models
    for name, factory, dtype in oof_dense_models:
        model = factory()
        model.fit(X_dense_train, y_train)
        full_models[name] = (model, 'dense')
    
    # Get validation predictions from base models
    val_predictions = []
    
    for name, (model, dtype) in full_models.items():
        if dtype == 'sparse':
            pred = model.predict(X_txt_val)
        else:
            pred = model.predict(X_dense_val)
        val_predictions.append(pred)
    
    val_predictions = np.column_stack(val_predictions)
    
    # Get meta-model predictions
    meta_predictions = {}
    for meta_name, meta_model in meta_models.items():
        meta_predictions[meta_name] = meta_model.predict(val_predictions)
    
    # Optimize meta-ensemble weights
    print("Optimizing meta-ensemble...")
    best_weights = None
    best_smape = float('inf')
    best_pred = None
    
    meta_names = list(meta_predictions.keys())
    weight_options = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    search_count = 0
    for w1 in weight_options:
        for w2 in weight_options:
            w3 = 1.0 - (w1 + w2)
            if w3 < 0 or w3 > 1:
                continue
            
            weights = [w1, w2, w3]
            
            # Weighted combination
            pred_log_blend = sum(w * meta_predictions[meta_name] for w, meta_name in zip(weights, meta_names))
            pred_price_try = np.expm1(pred_log_blend)
            pred_price_try = np.clip(pred_price_try, q_low * 0.6, q_high * 1.3)
            
            s_try = smape(y_val_true, pred_price_try)
            search_count += 1
            
            if s_try < best_smape:
                best_smape = s_try
                best_weights = weights
                best_pred = pred_price_try
                print(f"  New best meta-SMAPE: {s_try:.3f}% (search #{search_count})")
    
    print(f"Completed {search_count} meta-weight combinations")
    
    s = best_smape
    pred_price = best_pred
    corr = np.corrcoef(y_val_true, pred_price)[0, 1]
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=== Meta-Learning Stacking Results ===")
    weight_str = ", ".join([f"{name}:{w:.2f}" for name, w in zip(meta_names, best_weights)])
    print(f"Meta-model weights: {weight_str}")
    print(f"SMAPE: {s:.2f}%")
    print(f"Correlation: {corr:.4f}")
    print(f"Pred mean: ${pred_price.mean():.2f}")
    print(f"Actual mean: ${y_val_true.mean():.2f}")
    print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    artifacts = {
        'fx': fx, 'poly': poly, 'scaler': scaler, 'important_cols': important_cols,
        'vectorizer_word': vectorizer_word, 'vectorizer_char': vectorizer_char,
        'svd': svd, 'full_models': full_models, 'meta_models': meta_models,
        'meta_names': meta_names, 'meta_weights': best_weights,
        'clip_low': float(q_low), 'clip_high': float(q_high)
    }
    
    return s, artifacts

def predict_meta_test(test_csv: str, artifacts: dict, out_path: str):
    """Generate test predictions using meta-learning model"""
    print("Generating meta-learning test predictions...")
    test_df = pd.read_csv(test_csv)
    
    # Extract features
    fx = artifacts['fx']
    X_features_test = fx.transform(test_df)
    
    # Add polynomial interactions
    poly = artifacts['poly']
    important_cols = artifacts['important_cols']
    
    if poly and important_cols:
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
    
    # Get base model predictions
    base_predictions = []
    full_models = artifacts['full_models']
    
    for name, (model, dtype) in full_models.items():
        if dtype == 'sparse':
            pred = model.predict(X_txt_test)
        else:
            pred = model.predict(X_dense_test)
        base_predictions.append(pred)
    
    base_predictions = np.column_stack(base_predictions)
    
    # Get meta-model predictions
    meta_models = artifacts['meta_models']
    meta_names = artifacts['meta_names']
    meta_predictions = {}
    
    for meta_name in meta_names:
        meta_predictions[meta_name] = meta_models[meta_name].predict(base_predictions)
    
    # Final meta-ensemble
    meta_weights = artifacts['meta_weights']
    pred_log_blend = sum(w * meta_predictions[meta_name] for w, meta_name in zip(meta_weights, meta_names))
    pred_price = np.expm1(pred_log_blend)
    pred_price = np.clip(pred_price, artifacts['clip_low'], artifacts['clip_high'])
    
    # Save predictions
    out = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': np.round(pred_price, 2)})
    out.to_csv(out_path, index=False)
    print(f"Meta-learning predictions saved to {out_path}")

if __name__ == "__main__":
    print("🧠 Meta-Learning Stacking Model")
    print("Advanced Second-Level Learning with Comprehensive Features")
    print("Target: Push SMAPE from 50.45% to <44%")
    print("Using ONLY existing datasets + comprehensive image URL analysis")
    print("=" * 70)
    
    smape_val, art = train_meta_learning_model('dataset/train.csv', sample_size=45000, val_size=5000)
    
    if smape_val < 55.0:
        predict_meta_test('dataset/test.csv', art, 'dataset/test_out_meta_learning.csv')
    
    print(f"\nFinal Meta-Learning SMAPE: {smape_val:.2f}%")
    if smape_val < 44.0:
        print("🎉🎉🎉 TARGET ACHIEVED! <44% SMAPE 🎉🎉🎉")
        print("🏆 META-LEARNING SUCCESS! 🏆")
    else:
        gap = smape_val - 44.0
        print(f"Gap to target: {gap:.2f} percentage points")
        
        if smape_val < 48.0:
            print("🚀 Excellent improvement! Very close to target.")
        elif smape_val < 50.0:
            print("📈 Good improvement! Consider ensemble diversity tweaks.")
        else:
            print("🔍 Consider advanced regularization or feature selection.")