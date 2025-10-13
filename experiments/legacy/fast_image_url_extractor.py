"""
Fast Image URL Feature Extraction for Product Price Prediction
Extract features from image URLs without network requests
Next step towards <44% target from current 53.60%
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class FastImageURLExtractor:
    """Extract features from image URLs only (no network requests)"""
    
    def __init__(self):
        # Common patterns that correlate with product value
        self.quality_keywords = [
            'hd', 'hq', 'high', 'quality', 'resolution', 'dpi', 'retina',
            'premium', 'professional', 'studio', 'detailed'
        ]
        
        self.platform_patterns = {
            'amazon': ['amazon', 'amzn', 'a.co'],
            'shopify': ['shopify', 'cdn.shopify'],
            'cloudinary': ['cloudinary', 'res.cloudinary'],
            'imgur': ['imgur', 'i.imgur'],
            'wordpress': ['wordpress', 'wp-content'],
            'squarespace': ['squarespace', 'static1.squarespace'],
            'wix': ['wix', 'static.wix'],
            'cdn': ['cdn', 'static', 'assets', 'media']
        }
        
        self.image_formats = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'svg']
        
    def extract_url_features(self, image_url: str) -> Dict[str, float]:
        """Extract comprehensive features from image URL structure"""
        features = {}
        
        if pd.isna(image_url) or not isinstance(image_url, str) or len(image_url) < 10:
            return self._default_features()
        
        try:
            url_lower = image_url.lower()
            parsed = urlparse(image_url)
            
            # Basic URL structure
            features['url_length'] = min(len(image_url), 500)  # Cap outliers
            features['domain_length'] = len(parsed.netloc)
            features['path_length'] = len(parsed.path)
            features['path_segments'] = len([x for x in parsed.path.split('/') if x])
            features['has_query'] = 1 if parsed.query else 0
            features['has_fragment'] = 1 if parsed.fragment else 0
            
            # Security and quality indicators
            features['is_https'] = 1 if image_url.startswith('https') else 0
            features['has_ssl_indicator'] = 1 if 'ssl' in url_lower else 0
            
            # Platform detection (affects image quality/trust)
            for platform, keywords in self.platform_patterns.items():
                features[f'platform_{platform}'] = 1 if any(kw in url_lower for kw in keywords) else 0
            
            # Image format detection
            detected_format = None
            for fmt in self.image_formats:
                if f'.{fmt}' in url_lower:
                    features[f'format_{fmt}'] = 1
                    detected_format = fmt
                else:
                    features[f'format_{fmt}'] = 0
            
            # Format quality ranking (PNG > JPEG > WebP > others)
            format_quality_map = {'png': 0.9, 'jpeg': 0.8, 'jpg': 0.8, 'webp': 0.7, 'gif': 0.4, 'bmp': 0.3, 'svg': 0.6}
            features['format_quality_score'] = format_quality_map.get(detected_format, 0.5)
            
            # Dimension extraction from URL
            dimensions = self._extract_dimensions_from_url(image_url)
            features['url_width'] = dimensions.get('width', 0)
            features['url_height'] = dimensions.get('height', 0) 
            features['url_max_dimension'] = max(dimensions.get('width', 0), dimensions.get('height', 0))
            features['url_has_dimensions'] = 1 if features['url_max_dimension'] > 0 else 0
            
            # Size category based on URL hints
            max_dim = features['url_max_dimension']
            features['size_tiny'] = 1 if 0 < max_dim <= 100 else 0
            features['size_small'] = 1 if 100 < max_dim <= 300 else 0
            features['size_medium'] = 1 if 300 < max_dim <= 800 else 0
            features['size_large'] = 1 if 800 < max_dim <= 1500 else 0
            features['size_huge'] = 1 if max_dim > 1500 else 0
            
            # Quality indicators in URL
            features['quality_keywords'] = sum(1 for kw in self.quality_keywords if kw in url_lower)
            features['has_quality_indicator'] = 1 if features['quality_keywords'] > 0 else 0
            
            # File naming patterns (professional vs amateur)
            filename = parsed.path.split('/')[-1] if parsed.path else ''
            features['filename_length'] = len(filename)
            features['has_underscore'] = 1 if '_' in filename else 0
            features['has_dash'] = 1 if '-' in filename else 0
            features['has_numbers'] = 1 if any(c.isdigit() for c in filename) else 0
            features['is_hash_filename'] = 1 if len(filename) > 20 and not any(c.isalpha() for c in filename.replace('.', '').replace('-', '').replace('_', '')) else 0
            
            # URL complexity (more complex = more professional)
            features['complexity_score'] = min(
                (features['path_segments'] * 0.3 + 
                 features['has_query'] * 0.2 + 
                 features['is_https'] * 0.2 + 
                 features['quality_keywords'] * 0.3), 5.0
            )
            
            # CDN usage (indicates professional setup)
            cdn_indicators = ['cdn', 'static', 'assets', 'media', 'images', 'img', 'pics']
            features['uses_cdn'] = 1 if any(ind in url_lower for ind in cdn_indicators) else 0
            
            # Thumbnail indicators (smaller images, lower prices)
            thumb_indicators = ['thumb', 'thumbnail', 'small', 'mini', 'preview', 'icon']
            features['is_thumbnail'] = 1 if any(ind in url_lower for ind in thumb_indicators) else 0
            
            # Professional photography indicators
            pro_indicators = ['studio', 'professional', 'product', 'catalog', 'gallery', 'portfolio']
            features['professional_photo'] = 1 if any(ind in url_lower for ind in pro_indicators) else 0
            
            return features
            
        except Exception:
            return self._default_features()
    
    def _extract_dimensions_from_url(self, url: str) -> Dict[str, int]:
        """Extract image dimensions from URL patterns"""
        dimensions = {'width': 0, 'height': 0}
        
        # Common dimension patterns in URLs
        patterns = [
            r'(\d+)x(\d+)',           # 800x600
            r'_(\d+)_(\d+)',          # _800_600
            r'/(\d+)/(\d+)/',         # /800/600/
            r'w(\d+)h(\d+)',          # w800h600
            r'width(\d+)height(\d+)', # width800height600
            r'(\d+)-(\d+)',           # 800-600
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, url)
            for match in matches:
                try:
                    w, h = int(match[0]), int(match[1])
                    if 10 <= w <= 5000 and 10 <= h <= 5000:  # Reasonable bounds
                        dimensions['width'] = max(dimensions['width'], w)
                        dimensions['height'] = max(dimensions['height'], h)
                except (ValueError, IndexError):
                    continue
        
        # Single dimension patterns
        single_patterns = [
            r'w(\d+)', r'width(\d+)', r'size(\d+)', r'res(\d+)'
        ]
        
        for pattern in single_patterns:
            matches = re.findall(pattern, url)
            for match in matches:
                try:
                    dim = int(match)
                    if 10 <= dim <= 5000:
                        if dimensions['width'] == 0:
                            dimensions['width'] = dim
                        if dimensions['height'] == 0:
                            dimensions['height'] = dim
                except ValueError:
                    continue
        
        return dimensions
    
    def _default_features(self) -> Dict[str, float]:
        """Default features when URL is invalid"""
        features = {
            'url_length': 0, 'domain_length': 0, 'path_length': 0, 'path_segments': 0,
            'has_query': 0, 'has_fragment': 0, 'is_https': 0, 'has_ssl_indicator': 0,
            'url_width': 0, 'url_height': 0, 'url_max_dimension': 0, 'url_has_dimensions': 0,
            'size_tiny': 0, 'size_small': 0, 'size_medium': 1, 'size_large': 0, 'size_huge': 0,
            'quality_keywords': 0, 'has_quality_indicator': 0, 'format_quality_score': 0.5,
            'filename_length': 0, 'has_underscore': 0, 'has_dash': 0, 'has_numbers': 0, 'is_hash_filename': 0,
            'complexity_score': 0, 'uses_cdn': 0, 'is_thumbnail': 0, 'professional_photo': 0
        }
        
        # Platform features
        for platform in self.platform_patterns.keys():
            features[f'platform_{platform}'] = 0
        
        # Format features
        for fmt in self.image_formats:
            features[f'format_{fmt}'] = 0
        
        return features

def analyze_image_url_features(train_file: str, n_samples: int = 10000):
    """Analyze image URL features for correlation with price"""
    print(f"Analyzing image URL features from {n_samples} samples...")
    
    # Load data
    df = pd.read_csv(train_file, nrows=n_samples)
    extractor = FastImageURLExtractor()
    
    # Extract features
    features_list = []
    for i, row in df.iterrows():
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(df)} URLs...")
        
        features = extractor.extract_url_features(row.get('image_link', ''))
        features['sample_id'] = row.get('sample_id', i)
        features['price'] = row.get('price', 0)
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Analyze correlations
    feature_cols = [col for col in features_df.columns if col not in ['sample_id', 'price']]
    correlations = {}
    
    for col in feature_cols:
        try:
            corr = features_df[col].corr(features_df['price'])
            if not pd.isna(corr):
                correlations[col] = abs(corr)
        except:
            pass
    
    # Results
    print(f"\n=== Image URL Feature Analysis ===")
    print(f"Extracted {len(feature_cols)} features from {len(features_df)} samples")
    print(f"Price range: ${features_df['price'].min():.2f} - ${features_df['price'].max():.2f}")
    
    print(f"\nTop Image URL Features by Price Correlation:")
    print("-" * 45)
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for feature, corr in sorted_corrs[:20]:
        print(f"{feature:<30}: {corr:.4f}")
    
    # Save features
    output_file = 'image_url_features.csv'
    features_df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to: {output_file}")
    
    return features_df, correlations

if __name__ == "__main__":
    print("Fast Image URL Feature Extraction")
    print("=" * 40)
    
    # Analyze image URL patterns
    features_df, correlations = analyze_image_url_features('dataset/train.csv', n_samples=15000)
    
    print(f"\n=== Summary ===")
    print(f"Ready to integrate image URL features with main model!")
    print(f"Expected SMAPE improvement: 1-3 percentage points")
    print(f"Next: Combine with text+numeric model to push towards <44% target")