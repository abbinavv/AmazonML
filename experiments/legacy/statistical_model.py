"""
Statistical Learning Model - Direct Pattern Matching
Uses statistical analysis of training data for optimal SMAPE
"""

import os
import re
import pandas as pd
import numpy as np
import math

def analyze_training_patterns():
    """Analyze training data to extract optimal pricing patterns"""
    print("Analyzing training data patterns...")
    
    # Load sufficient training data
    train_data = pd.read_csv('dataset/train.csv', nrows=20000)
    
    # Basic statistics
    price_stats = {
        'mean': train_data['price'].mean(),
        'median': train_data['price'].median(),
        'std': train_data['price'].std(),
        'p10': train_data['price'].quantile(0.1),
        'p25': train_data['price'].quantile(0.25),
        'p75': train_data['price'].quantile(0.75),
        'p90': train_data['price'].quantile(0.9)
    }
    
    print(f"Price statistics: {price_stats}")
    
    # Text length analysis
    train_data['text_length'] = train_data['catalog_content'].str.len()
    
    # Create price bins for pattern analysis
    train_data['price_bin'] = pd.cut(train_data['price'], 
                                   bins=[0, 5, 10, 20, 35, 50, 100, float('inf')],
                                   labels=['very_low', 'low', 'medium_low', 'medium', 'medium_high', 'high', 'very_high'])
    
    # Analyze patterns by price bin
    patterns = {}
    for bin_name in train_data['price_bin'].unique():
        if pd.isna(bin_name):
            continue
        
        bin_data = train_data[train_data['price_bin'] == bin_name]
        patterns[bin_name] = {
            'avg_price': bin_data['price'].mean(),
            'avg_text_length': bin_data['text_length'].mean(),
            'sample_count': len(bin_data),
            'common_words': []
        }
        
        # Find common words in this price range
        all_text = ' '.join(bin_data['catalog_content'].fillna('').astype(str).str.lower())
        words = re.findall(r'\\b\\w+\\b', all_text)
        word_freq = pd.Series(words).value_counts()
        patterns[bin_name]['common_words'] = word_freq.head(20).index.tolist()
    
    print(f"Price bin patterns: {patterns}")
    return price_stats, patterns, train_data

class StatisticalPricingModel:
    def __init__(self, price_stats, patterns):
        self.price_stats = price_stats
        self.patterns = patterns
        
        # Create simplified category mapping based on common words
        self.category_keywords = {
            'jewelry': ['jewelry', 'ring', 'necklace', 'watch', 'gold', 'silver', 'diamond'],
            'automotive': ['car', 'auto', 'vehicle', 'tire', 'oil', 'brake'],
            'electronics': ['electronic', 'device', 'smart', 'tech', 'digital'],
            'health': ['health', 'vitamin', 'supplement', 'organic'],
            'food': ['food', 'snack', 'gourmet', 'cooking'],
            'tools': ['tool', 'drill', 'equipment', 'hardware'],
            'home': ['home', 'kitchen', 'furniture', 'house'],
            'clothing': ['clothing', 'shirt', 'dress', 'fashion']
        }
        
        # Price ranges by category (from analysis)
        self.category_price_ranges = {
            'jewelry': (8, 45),
            'automotive': (5, 40),
            'electronics': (6, 35),
            'health': (4, 30),
            'food': (3, 25),
            'tools': (8, 40),
            'home': (5, 35),
            'clothing': (6, 30),
            'other': (4, 25)
        }
    
    def classify_text_by_patterns(self, text):
        """Classify text based on learned patterns"""
        if not isinstance(text, str):
            return 'medium_low', 0.5
        
        text_lower = text.lower()
        text_length = len(text)
        
        # Score against each price bin pattern
        bin_scores = {}
        for bin_name, pattern in self.patterns.items():
            score = 0
            
            # Text length similarity
            length_diff = abs(text_length - pattern['avg_text_length'])
            length_score = max(0, 1 - length_diff / 2000)  # Normalize
            score += length_score * 0.3
            
            # Word matching
            common_words = pattern['common_words'][:10]  # Top 10 words
            word_matches = sum(1 for word in common_words if word in text_lower)
            word_score = word_matches / len(common_words) if common_words else 0
            score += word_score * 0.7
            
            bin_scores[bin_name] = score
        
        # Find best matching bin
        if bin_scores:
            best_bin = max(bin_scores, key=bin_scores.get)
            confidence = bin_scores[best_bin]
            return best_bin, confidence
        
        return 'medium_low', 0.5
    
    def detect_category(self, text):
        """Detect category for additional price guidance"""
        if not isinstance(text, str):
            return 'other'
        
        text_lower = text.lower()
        best_category = 'other'
        max_matches = 0
        
        for category, keywords in self.category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        return best_category
    
    def predict_price(self, sample_id, catalog_content, image_link):
        """Predict price using statistical patterns"""
        try:
            # Classify text by learned patterns
            price_bin, confidence = self.classify_text_by_patterns(catalog_content)
            
            # Get base price from pattern
            if price_bin in self.patterns:
                base_price = self.patterns[price_bin]['avg_price']
            else:
                base_price = self.price_stats['median']
            
            # Get category for additional guidance
            category = self.detect_category(catalog_content)
            min_price, max_price = self.category_price_ranges.get(category, (4, 25))
            
            # Adjust based on confidence and additional features
            text = str(catalog_content) if catalog_content else ""
            
            # Length adjustment
            if len(text) > 1500:
                base_price *= 1.1
            elif len(text) < 300:
                base_price *= 0.9
            
            # Premium indicators
            premium_words = ['premium', 'luxury', 'professional', 'authentic']
            premium_count = sum(1 for word in premium_words if word in text.lower())
            if premium_count > 0:
                base_price *= (1 + premium_count * 0.15)
            
            # Quantity indicators (bulk discount)
            quantity_patterns = [r'pack of (\\d+)', r'(\\d+)\\s*pack', r'set of (\\d+)']
            quantity = 1
            for pattern in quantity_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    quantity = int(match.group(1))
                    break
            
            if quantity > 1:
                base_price *= (1 / (1 + math.log(quantity) * 0.1))
            
            # Apply confidence weighting
            if confidence < 0.3:
                # Low confidence, move toward median
                base_price = base_price * 0.7 + self.price_stats['median'] * 0.3
            
            # Sample-specific variation
            hash_val = hash(str(sample_id)) % 100
            variation = (hash_val - 50) / 1000  # ±5% variation
            base_price *= (1 + variation)
            
            # Ensure bounds
            final_price = max(min_price, min(max_price, base_price))
            
            return round(final_price, 2)
            
        except Exception:
            return round(self.price_stats['median'], 2)

def main():
    """Main execution"""
    print("Statistical Learning Pricing Model")
    print("=" * 40)
    
    # Analyze training data
    price_stats, patterns, train_data = analyze_training_patterns()
    
    # Create model
    model = StatisticalPricingModel(price_stats, patterns)
    
    # Test on validation subset
    print("\\nTesting model on validation data...")
    val_data = train_data.tail(2000).copy()  # Use last 2000 as validation
    
    predictions = []
    actuals = []
    
    for _, row in val_data.iterrows():
        pred = model.predict_price(row['sample_id'], row['catalog_content'], row['image_link'])
        predictions.append(pred)
        actuals.append(row['price'])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate SMAPE
    smape = np.mean(np.abs(predictions - actuals) / ((np.abs(actuals) + np.abs(predictions)) / 2)) * 100
    
    print(f"\\nValidation Results:")
    print(f"SMAPE: {smape:.4f}%")
    print(f"Prediction mean: ${predictions.mean():.2f}")
    print(f"Actual mean: ${actuals.mean():.2f}")
    print(f"Prediction median: ${np.median(predictions):.2f}")
    print(f"Actual median: ${np.median(actuals):.2f}")
    
    # Generate test predictions
    print("\\nGenerating test predictions...")
    test = pd.read_csv('dataset/test.csv')
    
    test_predictions = []
    for i, row in test.iterrows():
        if i % 10000 == 0:
            print(f"Processing {i}/{len(test)}...")
        
        pred = model.predict_price(row['sample_id'], row['catalog_content'], row['image_link'])
        test_predictions.append(pred)
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test['sample_id'],
        'price': test_predictions
    })
    
    # Save results
    output_filename = 'dataset/test_out_statistical.csv'
    output_df.to_csv(output_filename, index=False)
    
    print(f"\\nStatistical predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Price range: ${output_df['price'].min():.2f} - ${output_df['price'].max():.2f}")
    print(f"Mean price: ${output_df['price'].mean():.2f}")
    print(f"Median price: ${output_df['price'].median():.2f}")
    
    print("\\nSample predictions:")
    print(output_df.head(10))
    
    return smape, output_df

if __name__ == "__main__":
    smape_score, predictions_df = main()
    print(f"\\nFinal SMAPE: {smape_score:.4f}%")
    print("Model ready for submission!")