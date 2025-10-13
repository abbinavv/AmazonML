"""
Recreate image cache using the original approach:
- URL feature extraction 
- Partial image downloads for metadata
- No full image downloads (much faster and SSL-safe)
"""

import pandas as pd
import numpy as np
import os
import json
from less_optimized_models.image_feature_extractor import ImageFeatureExtractor

def recreate_image_cache():
    """Recreate image cache with metadata approach"""
    
    # Create cache directory
    cache_dir = 'image_cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize feature extractor
    extractor = ImageFeatureExtractor()
    
    # Get all image URLs from datasets
    image_urls = set()
    for dataset in ['dataset/train.csv', 'dataset/test.csv']:
        try:
            df = pd.read_csv(dataset)
            if 'image_link' in df.columns:
                urls = df['image_link'].dropna().unique()
                image_urls.update(urls)
                print(f"Found {len(urls)} URLs in {dataset}")
        except Exception as e:
            print(f"Could not read {dataset}: {e}")
    
    print(f"Total unique image URLs: {len(image_urls)}")
    
    # Extract features for each URL and cache them
    cached_features = {}
    success_count = 0
    
    for i, url in enumerate(image_urls):
        try:
            # Extract URL features (fast, no download)
            url_features = extractor.extract_url_features(url)
            
            # Try to get image metadata (partial download)
            try:
                image_features = extractor.extract_image_features(url, timeout=5)
                features = {**url_features, **image_features}
            except:
                # If image download fails, use only URL features
                features = url_features
            
            # Cache the features
            url_hash = str(hash(url))
            cached_features[url_hash] = {
                'url': url,
                'features': features
            }
            
            success_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_urls)} URLs, {success_count} successful")
                
        except Exception as e:
            print(f"Failed to process {url}: {e}")
    
    # Save cached features to file
    cache_file = os.path.join(cache_dir, 'image_features.json')
    with open(cache_file, 'w') as f:
        json.dump(cached_features, f)
    
    print(f"Cached features for {success_count} images in {cache_file}")
    return cached_features

if __name__ == "__main__":
    recreate_image_cache()