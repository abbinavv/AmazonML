import pandas as pd
import numpy as np
import re
from collections import Counter

def analyze_training_data():
    """Analyze training data to understand patterns for better SMAPE score"""
    print("=== TRAINING DATA ANALYSIS ===")
    
    # Read training data sample
    print("Loading training data sample...")
    train_sample = pd.read_csv('dataset/train.csv', nrows=5000)
    print(f"Sample shape: {train_sample.shape}")
    print(f"Columns: {train_sample.columns.tolist()}")
    
    # Price analysis
    print("\n--- PRICE DISTRIBUTION ---")
    print(train_sample['price'].describe())
    
    print("\nPrice percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = train_sample['price'].quantile(p/100)
        print(f"{p:2d}th percentile: ${val:8.2f}")
    
    # Text length analysis
    print("\n--- TEXT ANALYSIS ---")
    train_sample['text_length'] = train_sample['catalog_content'].str.len()
    train_sample['word_count'] = train_sample['catalog_content'].str.split().str.len()
    
    print("Text length stats:")
    print(train_sample['text_length'].describe())
    print("\nWord count stats:")
    print(train_sample['word_count'].describe())
    
    # Correlation analysis
    print("\n--- CORRELATION ANALYSIS ---")
    print(f"Price vs Text Length correlation: {train_sample['price'].corr(train_sample['text_length']):.4f}")
    print(f"Price vs Word Count correlation: {train_sample['price'].corr(train_sample['word_count']):.4f}")
    
    # Category analysis
    print("\n--- CATEGORY ANALYSIS ---")
    categories = {
        'electronics': ['electronic', 'device', 'tech', 'digital', 'smart', 'bluetooth'],
        'jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'gold', 'silver'],
        'clothing': ['shirt', 'pants', 'dress', 'clothing', 'apparel', 'fashion'],
        'tools': ['tool', 'drill', 'equipment', 'hardware'],
        'automotive': ['car', 'auto', 'vehicle', 'tire'],
        'food': ['food', 'snack', 'gourmet', 'cooking'],
        'health': ['health', 'vitamin', 'supplement'],
        'home': ['home', 'kitchen', 'furniture']
    }
    
    for category, keywords in categories.items():
        mask = train_sample['catalog_content'].str.lower().str.contains('|'.join(keywords), na=False)
        if mask.sum() > 0:
            avg_price = train_sample[mask]['price'].mean()
            count = mask.sum()
            print(f"{category:12s}: {count:4d} items, avg price: ${avg_price:7.2f}")
    
    # Premium vs budget analysis
    print("\n--- PREMIUM vs BUDGET ANALYSIS ---")
    premium_keywords = ['premium', 'luxury', 'deluxe', 'professional', 'high-quality']
    budget_keywords = ['budget', 'affordable', 'economy', 'basic', 'discount']
    
    premium_mask = train_sample['catalog_content'].str.lower().str.contains('|'.join(premium_keywords), na=False)
    budget_mask = train_sample['catalog_content'].str.lower().str.contains('|'.join(budget_keywords), na=False)
    
    if premium_mask.sum() > 0:
        print(f"Premium items: {premium_mask.sum()}, avg price: ${train_sample[premium_mask]['price'].mean():.2f}")
    if budget_mask.sum() > 0:
        print(f"Budget items: {budget_mask.sum()}, avg price: ${train_sample[budget_mask]['price'].mean():.2f}")
    
    # Quantity analysis
    print("\n--- QUANTITY ANALYSIS ---")
    quantity_patterns = [r'pack of (\d+)', r'(\d+)\s*pack', r'(\d+)\s*count', r'value:\s*(\d+\.?\d*)']
    quantities = []
    
    for _, row in train_sample.iterrows():
        text = str(row['catalog_content']).lower()
        for pattern in quantity_patterns:
            match = re.search(pattern, text)
            if match:
                quantities.append((float(match.group(1)), row['price']))
                break
        else:
            quantities.append((1.0, row['price']))
    
    if quantities:
        qty_df = pd.DataFrame(quantities, columns=['quantity', 'price'])
        print("Quantity vs Price correlation:")
        print(f"Correlation: {qty_df['quantity'].corr(qty_df['price']):.4f}")
        
        # Group by quantity ranges
        qty_df['qty_range'] = pd.cut(qty_df['quantity'], bins=[0, 1, 2, 5, 10, float('inf')], 
                                    labels=['1', '2', '3-5', '6-10', '10+'])
        qty_summary = qty_df.groupby('qty_range')['price'].agg(['count', 'mean', 'std']).round(2)
        print("\nPrice by quantity range:")
        print(qty_summary)
    
    return train_sample

def calculate_smape(actual, predicted):
    """Calculate SMAPE score"""
    return np.mean(np.abs(predicted - actual) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100

if __name__ == "__main__":
    train_data = analyze_training_data()
    
    # Test current model on training sample
    print("\n=== CURRENT MODEL EVALUATION ===")
    from sample_code import predictor
    
    # Generate predictions for sample
    sample_size = min(1000, len(train_data))
    test_sample = train_data.head(sample_size)
    
    predictions = []
    for _, row in test_sample.iterrows():
        pred = predictor(row['sample_id'], row['catalog_content'], row['image_link'])
        predictions.append(pred)
    
    predictions = np.array(predictions)
    actual = test_sample['price'].values
    
    smape_score = calculate_smape(actual, predictions)
    print(f"Current model SMAPE on {sample_size} samples: {smape_score:.4f}%")
    
    # Basic statistics
    print(f"Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"Actual range: ${actual.min():.2f} - ${actual.max():.2f}")
    print(f"Prediction mean: ${predictions.mean():.2f}")
    print(f"Actual mean: ${actual.mean():.2f}")