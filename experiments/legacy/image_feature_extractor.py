"""
Image Feature Extraction for Product Price Prediction
Goal: Extract visual features from product image URLs to improve SMAPE
Next step towards <44% target from current 53.60%
"""

import pandas as pd
import numpy as np
import requests
from PIL import Image
import io
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

class ImageFeatureExtractor:
    """Extract features from product image URLs without downloading full images"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        # Common image characteristics that correlate with price
        self.size_thresholds = {
            'tiny': (0, 100),
            'small': (100, 300), 
            'medium': (300, 800),
            'large': (800, 1500),
            'huge': (1500, float('inf'))
        }
        
    def extract_url_features(self, image_url: str) -> Dict[str, float]:
        """Extract features from image URL itself (fast, no download)"""
        features = {}
        
        if pd.isna(image_url) or not isinstance(image_url, str):
            return self._default_url_features()
        
        try:
            # URL structure analysis
            parsed = urlparse(image_url)
            features['url_length'] = len(image_url)
            features['domain_length'] = len(parsed.netloc)
            features['path_segments'] = len([x for x in parsed.path.split('/') if x])
            
            # Quality indicators from URL
            url_lower = image_url.lower()
            features['has_https'] = 1 if image_url.startswith('https') else 0
            features['has_ssl'] = 1 if 'ssl' in url_lower else 0
            
            # Image format detection
            for fmt in ['jpg', 'jpeg', 'png', 'webp', 'gif']:
                features[f'format_{fmt}'] = 1 if fmt in url_lower else 0
            
            # Size hints in URL
            size_patterns = [
                r'(\d+)x(\d+)', r'_(\d+)_(\d+)', r'/(\d+)/(\d+)/',
                r'w(\d+)', r'h(\d+)', r'size(\d+)'
            ]
            max_dimension = 0
            for pattern in size_patterns:
                matches = re.findall(pattern, image_url)
                for match in matches:
                    if isinstance(match, tuple):
                        dims = [int(x) for x in match if x.isdigit()]
                        if dims:
                            max_dimension = max(max_dimension, max(dims))
                    elif match.isdigit():
                        max_dimension = max(max_dimension, int(match))
            
            features['url_max_dimension'] = min(max_dimension, 5000)  # Cap outliers
            
            # Quality/resolution indicators
            quality_terms = ['hq', 'hd', 'high', 'quality', 'resolution', 'dpi']
            features['quality_indicators'] = sum(1 for term in quality_terms if term in url_lower)
            
            # E-commerce platform detection
            platforms = {
                'amazon': ['amazon', 'amzn'],
                'shopify': ['shopify', 'cdn.shopify'],
                'cloudinary': ['cloudinary'],
                'imgur': ['imgur'],
                'other_cdn': ['cdn', 'static', 'images', 'img']
            }
            
            for platform, keywords in platforms.items():
                features[f'platform_{platform}'] = 1 if any(kw in url_lower for kw in keywords) else 0
            
            return features
            
        except Exception:
            return self._default_url_features()
    
    def extract_image_metadata(self, image_url: str, timeout: float = 3.0) -> Dict[str, float]:
        """Extract basic image metadata without full download (HEAD request + partial)"""
        features = {}
        
        if pd.isna(image_url) or not isinstance(image_url, str):
            return self._default_metadata_features()
        
        try:
            # Try HEAD request first for content info
            head_response = self.session.head(image_url, timeout=timeout, allow_redirects=True)
            
            if head_response.status_code == 200:
                content_length = head_response.headers.get('content-length')
                if content_length:
                    features['file_size_kb'] = min(int(content_length) / 1024, 10000)  # Cap at 10MB
                else:
                    features['file_size_kb'] = 0
                
                content_type = head_response.headers.get('content-type', '').lower()
                features['is_jpeg'] = 1 if 'jpeg' in content_type else 0
                features['is_png'] = 1 if 'png' in content_type else 0
                features['is_webp'] = 1 if 'webp' in content_type else 0
            
            # Try to get image dimensions with partial download
            try:
                # Download just enough to get image headers
                partial_response = self.session.get(
                    image_url, 
                    timeout=timeout, 
                    stream=True,
                    headers={'Range': 'bytes=0-2048'}  # First 2KB usually has dimensions
                )
                
                if partial_response.status_code in [200, 206]:
                    # Try to open with PIL to get dimensions
                    image_data = b''
                    for chunk in partial_response.iter_content(chunk_size=1024):
                        image_data += chunk
                        if len(image_data) >= 2048:
                            break
                    
                    try:
                        with Image.open(io.BytesIO(image_data)) as img:
                            width, height = img.size
                            features['image_width'] = min(width, 5000)
                            features['image_height'] = min(height, 5000) 
                            features['image_area'] = min(width * height, 25000000)
                            features['aspect_ratio'] = min(width / max(height, 1), 10)
                            
                            # Image size category
                            max_dim = max(width, height)
                            for size_name, (min_val, max_val) in self.size_thresholds.items():
                                features[f'size_{size_name}'] = 1 if min_val <= max_dim < max_val else 0
                            
                    except Exception:
                        # If PIL fails, use defaults
                        features.update(self._default_dimension_features())
                        
            except Exception:
                features.update(self._default_dimension_features())
            
            return features
            
        except Exception:
            return self._default_metadata_features()
    
    def extract_comprehensive_features(self, image_url: str) -> Dict[str, float]:
        """Extract all available image features"""
        features = {}
        
        # Fast URL-based features
        url_features = self.extract_url_features(image_url)
        features.update(url_features)
        
        # Metadata features (with network requests)
        metadata_features = self.extract_image_metadata(image_url)
        features.update(metadata_features)
        
        # Derived features
        if 'image_width' in features and 'image_height' in features:
            features['is_square'] = 1 if abs(features['image_width'] - features['image_height']) < 50 else 0
            features['is_portrait'] = 1 if features['image_height'] > features['image_width'] * 1.2 else 0
            features['is_landscape'] = 1 if features['image_width'] > features['image_height'] * 1.2 else 0
        
        # Quality score (composite)
        quality_score = 0
        quality_score += features.get('has_https', 0) * 0.2
        quality_score += features.get('quality_indicators', 0) * 0.3
        quality_score += min(features.get('file_size_kb', 0) / 100, 2) * 0.2  # Larger files = higher quality
        quality_score += min(features.get('image_area', 0) / 100000, 2) * 0.3  # Larger images = higher quality
        features['image_quality_score'] = min(quality_score, 5.0)
        
        return features
    
    def _default_url_features(self) -> Dict[str, float]:
        """Default URL features when extraction fails"""
        return {
            'url_length': 0, 'domain_length': 0, 'path_segments': 0,
            'has_https': 0, 'has_ssl': 0,
            'format_jpg': 0, 'format_jpeg': 0, 'format_png': 0, 'format_webp': 0, 'format_gif': 0,
            'url_max_dimension': 0, 'quality_indicators': 0,
            'platform_amazon': 0, 'platform_shopify': 0, 'platform_cloudinary': 0, 
            'platform_imgur': 0, 'platform_other_cdn': 0
        }
    
    def _default_metadata_features(self) -> Dict[str, float]:
        """Default metadata features when extraction fails"""
        features = {
            'file_size_kb': 0, 'is_jpeg': 0, 'is_png': 0, 'is_webp': 0
        }
        features.update(self._default_dimension_features())
        return features
    
    def _default_dimension_features(self) -> Dict[str, float]:
        """Default dimension features"""
        features = {
            'image_width': 300, 'image_height': 300, 'image_area': 90000, 'aspect_ratio': 1.0
        }
        # Set medium as default size
        for size_name in self.size_thresholds.keys():
            features[f'size_{size_name}'] = 1 if size_name == 'medium' else 0
        return features

def process_sample_images(csv_file: str, n_samples: int = 1000) -> pd.DataFrame:
    """Process a sample of images to test feature extraction"""
    print(f"Processing {n_samples} sample images from {csv_file}...")
    
    df = pd.read_csv(csv_file, nrows=n_samples)
    extractor = ImageFeatureExtractor()
    
    features_list = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"  Processed {i}/{len(df)} images...")
        
        try:
            features = extractor.extract_comprehensive_features(row.get('image_link', ''))
            features['sample_id'] = row.get('sample_id', i)
            if 'price' in row:
                features['price'] = row['price']
            features_list.append(features)
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            # Add default features
            features = extractor._default_url_features()
            features.update(extractor._default_metadata_features())
            features['sample_id'] = row.get('sample_id', i)
            if 'price' in row:
                features['price'] = row['price']
            features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    print(f"Extracted {len(features_df.columns)-1} image features")
    return features_df

if __name__ == "__main__":
    # Test with training data sample
    print("Image Feature Extraction for Product Pricing")
    print("=" * 50)
    
    # Process sample to understand image feature patterns
    train_features = process_sample_images('dataset/train.csv', n_samples=2000)
    
    # Save features for analysis
    train_features.to_csv('image_features_sample.csv', index=False)
    
    # Analyze correlation with price
    if 'price' in train_features.columns:
        feature_cols = [col for col in train_features.columns if col not in ['sample_id', 'price']]
        correlations = {}
        
        for col in feature_cols:
            try:
                corr = train_features[col].corr(train_features['price'])
                if not pd.isna(corr):
                    correlations[col] = abs(corr)
            except:
                pass
        
        print(f"\nTop Image Features by Price Correlation:")
        print("-" * 40)
        sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        for feature, corr in sorted_corrs[:15]:
            print(f"{feature:<25}: {corr:.4f}")
        
        print(f"\nImage feature extraction completed!")
        print(f"Features saved to: image_features_sample.csv")
        print(f"Ready to integrate with main model...")