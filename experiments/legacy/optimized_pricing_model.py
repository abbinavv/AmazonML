"""
Optimized Product Pricing Model for Best SMAPE Score
Based on training data analysis and insights
"""

import os
import re
import math
import pandas as pd
import numpy as np
from collections import defaultdict

class OptimizedPricingModel:
    def __init__(self):
        """Initialize with data-driven parameters from analysis"""
        
        # Base price calibrated to training data median
        self.base_price = 13.99  # Training data median
        
        # Category multipliers based on actual training data averages
        self.category_multipliers = {
            'jewelry': 1.31,      # 31.66 / 24.18 = 1.31
            'automotive': 1.28,   # 30.86 / 24.18 = 1.28
            'tools': 1.31,        # 31.70 / 24.18 = 1.31
            'health': 1.24,       # 29.94 / 24.18 = 1.24
            'electronics': 1.20,  # 28.97 / 24.18 = 1.20
            'clothing': 1.12,     # 27.14 / 24.18 = 1.12
            'home': 1.16,         # 28.09 / 24.18 = 1.16
            'food': 1.02,         # 24.73 / 24.18 = 1.02
            'books': 0.85,        # Estimated lower
            'toys': 0.90          # Estimated lower
        }
        
        # Premium/budget analysis shows budget items are actually more expensive
        # This suggests "budget" might indicate bulk/wholesale pricing
        self.sentiment_keywords = {
            'premium': ['premium', 'luxury', 'deluxe', 'professional', 'authentic', 'original'],
            'budget': ['budget', 'affordable', 'economy', 'basic', 'value', 'pack'],  # Pack often = bulk
            'high_end': ['gold', 'silver', 'platinum', 'diamond', 'leather', 'steel'],
            'brand': ['brand', 'trademark', '®', '™', 'official', 'certified']
        }
        
        # Price range adjustments based on text patterns
        self.price_indicators = {
            'expensive_words': ['diamond', 'platinum', 'luxury', 'premium', 'professional'],
            'cheap_words': ['plastic', 'basic', 'simple', 'economy'],
            'bulk_indicators': ['pack of', 'set of', 'bulk', 'wholesale', 'dozen']
        }
        
    def extract_numeric_features(self, text):
        """Extract numeric features with focus on price-relevant numbers"""
        if not isinstance(text, str):
            return {}
        
        # Find all numbers, focusing on reasonable price-related values
        numbers = re.findall(r'\d+\.?\d*', text)
        numbers = [float(x) for x in numbers if 0.1 <= float(x) <= 1000]  # Reasonable range
        
        if not numbers:
            return {'max_number': 0, 'number_count': 0, 'size_indicator': 1.0}
        
        # Size/dimension indicators (larger products might be more expensive)
        size_patterns = [r'(\d+)\s*(?:inch|in|cm|mm|oz|lb|kg|liter|ml)', 
                        r'(\d+)\s*x\s*(\d+)', r'(\d+)"\s*']
        
        size_numbers = []
        for pattern in size_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    size_numbers.extend([float(x) for x in match if x.replace('.', '').isdigit()])
                else:
                    size_numbers.append(float(match))
        
        return {
            'max_number': max(numbers),
            'number_count': len(numbers),
            'size_indicator': max(size_numbers) if size_numbers else 1.0
        }
    
    def detect_category(self, text):
        """Improved category detection with multi-keyword matching"""
        if not isinstance(text, str):
            return 'home'
        
        text_lower = text.lower()
        
        # Enhanced category keywords based on training data
        category_keywords = {
            'jewelry': ['jewelry', 'ring', 'necklace', 'bracelet', 'earring', 'watch', 
                       'pendant', 'chain', 'diamond', 'gold', 'silver', 'gemstone'],
            'automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'automotive', 
                          'motorcycle', 'truck', 'parts', 'oil', 'brake'],
            'tools': ['tool', 'drill', 'hammer', 'saw', 'wrench', 'equipment', 
                     'hardware', 'screwdriver', 'pliers'],
            'electronics': ['electronic', 'device', 'gadget', 'tech', 'digital', 
                           'smart', 'bluetooth', 'wifi', 'computer', 'phone'],
            'health': ['health', 'medical', 'vitamin', 'supplement', 'wellness', 
                      'medicine', 'care', 'treatment', 'therapy'],
            'clothing': ['shirt', 'pants', 'dress', 'jacket', 'clothing', 'apparel', 
                        'fashion', 'wear', 'fabric', 'cotton'],
            'home': ['home', 'house', 'kitchen', 'bedroom', 'living', 'furniture', 
                    'decor', 'appliance'],
            'food': ['food', 'snack', 'drink', 'beverage', 'cooking', 'recipe', 
                    'gourmet', 'organic', 'ingredient'],
            'books': ['book', 'novel', 'guide', 'manual', 'reading', 'literature', 
                     'textbook', 'author'],
            'toys': ['toy', 'game', 'play', 'kids', 'children', 'fun', 'puzzle', 'doll']
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, default to 'home'
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'home'
    
    def extract_quantity_info(self, text):
        """Enhanced quantity extraction with multiple patterns"""
        if not isinstance(text, str):
            return 1.0
        
        text_lower = text.lower()
        
        # Comprehensive quantity patterns
        quantity_patterns = [
            r'pack of (\d+)',
            r'(\d+)\s*pack',
            r'(\d+)\s*piece',
            r'(\d+)\s*count',
            r'set of (\d+)',
            r'(\d+)\s*items?',
            r'quantity[:\s]*(\d+)',
            r'value:\s*(\d+\.?\d*)',  # IPQ pattern
            r'(\d+)\s*per\s*pack',
            r'(\d+)\s*units?'
        ]
        
        for pattern in quantity_patterns:
            match = re.search(pattern, text_lower)
            if match:
                qty = float(match.group(1))
                # Cap at reasonable values
                return min(qty, 100)
        
        return 1.0
    
    def analyze_text_sentiment(self, text):
        """Improved sentiment analysis based on training insights"""
        if not isinstance(text, str):
            return {'premium_score': 0, 'brand_score': 0, 'material_score': 0}
        
        text_lower = text.lower()
        
        # Count different types of indicators
        premium_score = sum(1 for word in self.sentiment_keywords['premium'] if word in text_lower)
        brand_score = sum(1 for word in self.sentiment_keywords['brand'] if word in text_lower)
        
        # Material quality score
        material_words = ['gold', 'silver', 'platinum', 'diamond', 'titanium', 'steel', 'leather']
        material_score = sum(1 for material in material_words if material in text_lower)
        
        return {
            'premium_score': premium_score,
            'brand_score': brand_score, 
            'material_score': material_score
        }
    
    def predict_price(self, sample_id, catalog_content, image_link):
        """Optimized price prediction based on training data insights"""
        try:
            text = str(catalog_content) if catalog_content else ""
            
            # Start with data-driven base price
            base_price = self.base_price
            
            # Category-based adjustment (major factor)
            category = self.detect_category(text)
            category_factor = self.category_multipliers.get(category, 1.0)
            
            # Text length factor (calibrated to training correlation)
            text_length = len(text)
            # Moderate influence based on 0.16 correlation
            length_factor = 1 + (text_length - 897) / 5000  # Normalized around training mean
            length_factor = max(0.5, min(2.0, length_factor))  # Bounded
            
            # Sentiment analysis
            sentiment = self.analyze_text_sentiment(text)
            sentiment_factor = 1.0
            sentiment_factor += sentiment['premium_score'] * 0.15  # Premium boost
            sentiment_factor += sentiment['brand_score'] * 0.10    # Brand boost  
            sentiment_factor += sentiment['material_score'] * 0.20  # Material boost
            
            # Quantity adjustment (training shows negative correlation with bulk)
            quantity = self.extract_quantity_info(text)
            if quantity > 1:
                # Bulk discount effect observed in training data
                quantity_factor = 1 / (1 + math.log(quantity) * 0.1)
            else:
                quantity_factor = 1.0
            
            # Numeric features (size/dimension indicators)
            numeric_features = self.extract_numeric_features(text)
            size_factor = 1 + (numeric_features['size_indicator'] - 1) * 0.05
            size_factor = max(0.8, min(1.5, size_factor))
            
            # Image quality factor (minimal impact)
            image_factor = 1.0
            if isinstance(image_link, str) and image_link:
                if any(quality in image_link.lower() for quality in ['large', 'high', 'hd']):
                    image_factor = 1.02
            
            # Calculate intermediate price
            predicted_price = (base_price * 
                             category_factor * 
                             length_factor * 
                             sentiment_factor * 
                             quantity_factor * 
                             size_factor * 
                             image_factor)
            
            # Add controlled randomness to avoid identical predictions
            hash_factor = (hash(str(sample_id)) % 100) / 1000  # -0.05 to +0.05
            randomness = 0.95 + hash_factor
            predicted_price *= randomness
            
            # Final bounds based on training data (0.13 to 613.58)
            predicted_price = max(0.5, min(200.0, predicted_price))
            
            return round(predicted_price, 2)
            
        except Exception as e:
            # Robust fallback
            return round(self.base_price, 2)

# Global model instance
optimized_model = OptimizedPricingModel()

def predictor(sample_id, catalog_content, image_link):
    """
    Optimized predictor function for best SMAPE score
    """
    return optimized_model.predict_price(sample_id, catalog_content, image_link)

if __name__ == "__main__":
    print("Optimized Product Pricing Model")
    print("=" * 40)
    
    # Test on training sample first
    print("Testing on training sample...")
    train_sample = pd.read_csv('dataset/train.csv', nrows=1000)
    
    predictions = []
    actuals = []
    
    for _, row in train_sample.iterrows():
        pred = predictor(row['sample_id'], row['catalog_content'], row['image_link'])
        predictions.append(pred)
        actuals.append(row['price'])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate SMAPE
    smape = np.mean(np.abs(predictions - actuals) / ((np.abs(actuals) + np.abs(predictions)) / 2)) * 100
    
    print(f"SMAPE on training sample: {smape:.4f}%")
    print(f"Prediction mean: ${predictions.mean():.2f}")
    print(f"Actual mean: ${actuals.mean():.2f}")
    print(f"Prediction std: ${predictions.std():.2f}")
    print(f"Actual std: ${actuals.std():.2f}")
    
    # Generate test predictions
    print("\nGenerating test predictions...")
    test = pd.read_csv('dataset/test.csv')
    
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']), 
        axis=1
    )
    
    # Save predictions
    output_df = test[['sample_id', 'price']]
    output_filename = 'dataset/test_out_optimized.csv'
    output_df.to_csv(output_filename, index=False)
    
    print(f"Optimized predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Price range: ${output_df['price'].min():.2f} - ${output_df['price'].max():.2f}")
    print(f"Mean price: ${output_df['price'].mean():.2f}")
    print(f"Median price: ${output_df['price'].median():.2f}")
    
    print("\nSample predictions:")
    print(output_df.head(10))