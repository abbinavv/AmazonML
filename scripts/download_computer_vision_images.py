"""
Computer Vision Image Downloader
Downloads actual images from URLs for deep learning feature extraction
This recreates the original computer vision approach that downloads full images
"""

import pandas as pd
import numpy as np
import os
import requests
from pathlib import Path
from tqdm import tqdm
import time
from urllib.parse import urlparse
import ssl
import urllib3

# Disable SSL warnings for problematic certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def setup_ssl_context():
    """Setup SSL context to handle certificate issues"""
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context

def download_single_image(url, save_dir, timeout=10, max_retries=2):
    """Download a single image with error handling"""
    try:
        # Create filename from URL
        parsed = urlparse(url)
        filename = Path(parsed.path).name
        if not filename or '.' not in filename:
            filename = f"image_{abs(hash(url))}.jpg"
        
        save_path = os.path.join(save_dir, filename)
        
        # Skip if already exists
        if os.path.exists(save_path):
            return True, "Already exists"
        
        # Download with retries
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        for attempt in range(max_retries):
            try:
                response = session.get(url, timeout=timeout, verify=False, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'image' not in content_type:
                    return False, f"Not an image: {content_type}"
                
                # Download image
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return True, "Downloaded"
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return False, str(e)
                time.sleep(0.5)  # Brief delay before retry
        
    except Exception as e:
        return False, str(e)

def download_dataset_images():
    """Download all images from train.csv and test.csv"""
    
    # Create image directory
    image_dir = "image_cache"
    os.makedirs(image_dir, exist_ok=True)
    
    # Get all unique image URLs
    all_urls = set()
    datasets = ["dataset/train.csv", "dataset/test.csv"]
    
    print("Loading image URLs from datasets...")
    for dataset_path in datasets:
        try:
            df = pd.read_csv(dataset_path)
            if 'image_link' in df.columns:
                urls = df['image_link'].dropna().unique()
                all_urls.update(urls)
                print(f"Found {len(urls)} URLs in {dataset_path}")
        except Exception as e:
            print(f"Error reading {dataset_path}: {e}")
    
    print(f"Total unique image URLs: {len(all_urls)}")
    
    # Download images with progress bar
    successful = 0
    failed = 0
    
    print("Starting image download...")
    for url in tqdm(all_urls, desc="Downloading images"):
        try:
            success, message = download_single_image(url, image_dir)
            if success:
                successful += 1
            else:
                failed += 1
                if failed <= 10:  # Show first 10 failures
                    print(f"Failed to download {url}: {message}")
        except Exception as e:
            failed += 1
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Images saved in: {image_dir}")
    
    # List some downloaded files
    downloaded_files = os.listdir(image_dir)
    if downloaded_files:
        print(f"Sample files: {downloaded_files[:5]}")
    
    return successful, failed

if __name__ == "__main__":
    download_dataset_images()