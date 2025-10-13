"""
Advanced Training Data Analysis for <44% SMAPE
Comprehensive feature extraction and pattern discovery
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

def comprehensive_training_analysis():
    """Analyze full training data to extract powerful features"""
    print("=== COMPREHENSIVE TRAINING ANALYSIS ===")
    
    # Load larger sample
    print("Loading 50K training samples...")
    train_data = pd.read_csv('dataset/train.csv', nrows=50000)
    
    print(f"Loaded {len(train_data)} samples")
    print(f"Price range: ${train_data['price'].min():.2f} - ${train_data['price'].max():.2f}")
    print(f"Price statistics:")
    print(train_data['price'].describe())
    
    # Extract comprehensive features
    print("\nExtracting comprehensive features...")
    features_df = extract_all_features(train_data)
    
    # Analyze correlations
    print("\nAnalyzing feature correlations with price...")
    correlations = {}
    for col in features_df.columns:
        if col != 'price' and features_df[col].dtype in ['int64', 'float64']:
            corr = features_df[col].corr(features_df['price'])
            if not np.isnan(corr):
                correlations[col] = corr
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("Top correlations with price:")
    for feature, corr in sorted_corr[:15]:
        print(f"  {feature:25s}: {corr:7.4f}")
    
    return features_df, correlations

def extract_all_features(train_data):
    """Extract comprehensive features from training data"""
    features = train_data.copy()
    
    # Basic text features
    features['text_length'] = features['catalog_content'].str.len()
    features['word_count'] = features['catalog_content'].str.split().str.len()
    features['sentence_count'] = features['catalog_content'].str.count(r'[.!?]') + 1
    features['char_per_word'] = features['text_length'] / features['word_count']
    
    # Advanced text features
    print("Extracting text features...")
    features['has_bullet_points'] = features['catalog_content'].str.contains('bullet point|•', case=False, na=False).astype(int)
    features['description_sections'] = features['catalog_content'].str.lower().str.count('description:') + features['catalog_content'].str.lower().str.count('bullet point')
    features['capital_ratio'] = features['catalog_content'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
    
    # Numeric features from text
    print("Extracting numeric features...")
    features['numbers'] = features['catalog_content'].apply(extract_numbers_advanced)
    features['max_number'] = features['numbers'].apply(lambda x: max(x) if x else 0)
    features['min_number'] = features['numbers'].apply(lambda x: min(x) if x else 0)
    features['avg_number'] = features['numbers'].apply(lambda x: np.mean(x) if x else 0)
    features['number_count'] = features['numbers'].apply(len)
    features['number_range'] = features['max_number'] - features['min_number']
    
    # Quantity indicators
    print("Extracting quantity features...")
    features['quantity'] = features['catalog_content'].apply(extract_quantity_advanced)
    features['is_bulk'] = (features['quantity'] > 1).astype(int)
    features['pack_size_log'] = np.log1p(features['quantity'])
    
    # Category features
    print("Extracting category features...")
    category_features = extract_category_features(features['catalog_content'])
    for cat, scores in category_features.items():
        features[f'cat_{cat}'] = scores
    
    # Premium/quality indicators
    print("Extracting quality features...")
    quality_features = extract_quality_features(features['catalog_content'])
    for qual, scores in quality_features.items():
        features[f'qual_{qual}'] = scores
    
    # Brand indicators
    print("Extracting brand features...")
    brand_features = extract_brand_features(features['catalog_content'])
    for brand, scores in brand_features.items():
        features[f'brand_{brand}'] = scores
    
    # Size/dimension features
    print("Extracting size features...")
    size_features = extract_size_features(features['catalog_content'])
    for size_type, values in size_features.items():
        features[f'size_{size_type}'] = values
    
    # Price-related keywords
    print("Extracting price keywords...")
    price_keywords = extract_price_keywords(features['catalog_content'])
    for keyword_type, counts in price_keywords.items():
        features[f'price_{keyword_type}'] = counts
    
    return features

def extract_numbers_advanced(text):
    """Extract numbers with context awareness"""
    if pd.isna(text):
        return []
    
    text_str = str(text).lower()
    
    # Find numbers with context
    numbers = []
    
    # Dimensions (likely size indicators)
    dim_patterns = [r'(\d+\.?\d*)\s*(?:inch|in|cm|mm|ft|meter|yard)', 
                   r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)', 
                   r'(\d+\.?\d*)"\s*x\s*(\d+\.?\d*)"']
    
    for pattern in dim_patterns:
        matches = re.findall(pattern, text_str)
        for match in matches:
            if isinstance(match, tuple):
                numbers.extend([float(x) for x in match if x.replace('.', '').isdigit()])
            else:
                numbers.append(float(match))
    
    # Weights/volumes
    weight_patterns = [r'(\d+\.?\d*)\s*(?:oz|lb|kg|gram|pound)', 
                      r'(\d+\.?\d*)\s*(?:ml|liter|gallon|quart)']
    
    for pattern in weight_patterns:
        matches = re.findall(pattern, text_str)
        for match in matches:
            numbers.append(float(match))
    
    # General numbers (filtered for reasonable ranges)
    general_numbers = re.findall(r'\b(\d+\.?\d*)\b', text_str)
    for num_str in general_numbers:
        num = float(num_str)
        if 0.1 <= num <= 2000:  # Reasonable range
            numbers.append(num)
    
    return numbers

def extract_quantity_advanced(text):
    """Advanced quantity extraction"""
    if pd.isna(text):
        return 1.0
    
    text_lower = str(text).lower()
    
    # Comprehensive quantity patterns
    quantity_patterns = [
        r'pack of (\d+)', r'(\d+)\s*pack\b', r'(\d+)\s*piece', r'(\d+)\s*count\b',
        r'set of (\d+)', r'(\d+)\s*items?\b', r'quantity[:\s]*(\d+)',
        r'value:\s*(\d+\.?\d*)', r'(\d+)\s*per\s*pack', r'(\d+)\s*units?\b',
        r'(\d+)\s*pair', r'(\d+)\s*dozen', r'(\d+)x\s*pack', r'multipack\s*(\d+)',
        r'bulk\s*(\d+)', r'(\d+)\s*box', r'case\s*of\s*(\d+)'
    ]
    
    for pattern in quantity_patterns:
        match = re.search(pattern, text_lower)
        if match:
            qty = float(match.group(1))
            return min(qty, 100)  # Cap at 100
    
    # Check for bulk indicators without explicit numbers
    if any(word in text_lower for word in ['bulk', 'wholesale', 'case', 'dozen']):
        return 12.0  # Assume typical bulk size
    
    return 1.0

def extract_category_features(catalog_series):
    """Extract detailed category features"""
    categories = {
        'jewelry': ['jewelry', 'ring', 'necklace', 'bracelet', 'earring', 'watch', 'pendant', 'chain'],
        'automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'brake', 'oil', 'filter', 'part'],
        'electronics': ['electronic', 'device', 'gadget', 'smart', 'bluetooth', 'digital', 'tech', 'computer'],
        'tools': ['tool', 'drill', 'hammer', 'saw', 'wrench', 'equipment', 'hardware', 'socket'],
        'health': ['health', 'vitamin', 'supplement', 'medical', 'wellness', 'organic', 'natural'],
        'clothing': ['clothing', 'shirt', 'dress', 'pants', 'jacket', 'fashion', 'apparel', 'fabric'],
        'home': ['home', 'kitchen', 'furniture', 'decor', 'house', 'room', 'storage', 'appliance'],
        'food': ['food', 'snack', 'drink', 'beverage', 'cooking', 'gourmet', 'organic', 'ingredient'],
        'beauty': ['beauty', 'cosmetic', 'makeup', 'skincare', 'lotion', 'cream', 'fragrance'],
        'sports': ['sport', 'fitness', 'exercise', 'gym', 'outdoor', 'athletic', 'training'],
        'books': ['book', 'novel', 'guide', 'manual', 'reading', 'textbook', 'magazine'],
        'toys': ['toy', 'game', 'play', 'kids', 'children', 'fun', 'puzzle', 'doll']
    }
    
    category_scores = {}
    for category, keywords in categories.items():
        scores = catalog_series.str.lower().apply(
            lambda x: sum(1 for keyword in keywords if keyword in str(x))
        )
        category_scores[category] = scores
    
    return category_scores

def extract_quality_features(catalog_series):
    """Extract quality/premium indicators"""
    quality_indicators = {
        'premium': ['premium', 'luxury', 'deluxe', 'high-end', 'top-quality', 'superior'],
        'professional': ['professional', 'commercial', 'industrial', 'heavy-duty', 'pro-grade'],
        'authentic': ['authentic', 'genuine', 'original', 'real', 'certified', 'official'],
        'handmade': ['handmade', 'hand-crafted', 'artisan', 'custom', 'bespoke'],
        'durable': ['durable', 'long-lasting', 'sturdy', 'robust', 'heavy-duty', 'reinforced'],
        'budget': ['budget', 'affordable', 'economy', 'basic', 'standard', 'entry-level']
    }
    
    quality_scores = {}
    for quality, keywords in quality_indicators.items():
        scores = catalog_series.str.lower().apply(
            lambda x: sum(1 for keyword in keywords if keyword in str(x))
        )
        quality_scores[quality] = scores
    
    return quality_scores

def extract_brand_features(catalog_series):
    """Extract brand-related features"""
    brand_indicators = {
        'trademarked': ['®', '™', 'trademark', 'registered'],
        'branded': ['brand', 'company', 'corp', 'inc', 'ltd', 'llc'],
        'licensed': ['licensed', 'authorized', 'official', 'certified']
    }
    
    brand_scores = {}
    for brand_type, keywords in brand_indicators.items():
        scores = catalog_series.str.lower().apply(
            lambda x: sum(1 for keyword in keywords if keyword in str(x))
        )
        brand_scores[brand_type] = scores
    
    return brand_scores

def extract_size_features(catalog_series):
    """Extract size/dimension features"""
    size_features = {}
    
    # Size descriptors
    size_words = {
        'large': ['large', 'big', 'jumbo', 'oversized', 'xl', 'extra large'],
        'small': ['small', 'mini', 'compact', 'tiny', 'xs', 'petite'],
        'medium': ['medium', 'standard', 'regular', 'normal'],
        'king': ['king', 'queen', 'full', 'twin'],  # Common for household items
    }
    
    for size_type, keywords in size_words.items():
        scores = catalog_series.str.lower().apply(
            lambda x: sum(1 for keyword in keywords if keyword in str(x))
        )
        size_features[size_type] = scores
    
    return size_features

def extract_price_keywords(catalog_series):
    """Extract price-related keywords"""
    price_keywords = {
        'expensive': ['expensive', 'costly', 'pricey', 'high-priced'],
        'cheap': ['cheap', 'inexpensive', 'low-cost', 'bargain'],
        'value': ['value', 'deal', 'savings', 'discount', 'sale'],
        'exclusive': ['exclusive', 'limited', 'rare', 'special', 'unique']
    }
    
    keyword_scores = {}
    for keyword_type, keywords in price_keywords.items():
        scores = catalog_series.str.lower().apply(
            lambda x: sum(1 for keyword in keywords if keyword in str(x))
        )
        keyword_scores[keyword_type] = scores
    
    return keyword_scores

if __name__ == "__main__":
    features_df, correlations = comprehensive_training_analysis()
    
    print(f"\n=== FEATURE EXTRACTION SUMMARY ===")
    print(f"Total features extracted: {len(features_df.columns)}")
    print(f"Features with correlation > 0.1: {sum(1 for c in correlations.values() if abs(c) > 0.1)}")
    print(f"Features with correlation > 0.2: {sum(1 for c in correlations.values() if abs(c) > 0.2)}")
    
    # Save features for model building
    features_file = 'comprehensive_features.csv'
    features_df.to_csv(features_file, index=False)
    print(f"\nFeatures saved to {features_file}")
    
    # Save correlation analysis
    corr_df = pd.DataFrame(list(correlations.items()), columns=['feature', 'correlation'])
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
    corr_df.to_csv('feature_correlations.csv', index=False)
    print("Correlations saved to feature_correlations.csv")