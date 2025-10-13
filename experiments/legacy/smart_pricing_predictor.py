"""
Smart Product Pricing - Enhanced Predictor
A more sophisticated approach to price prediction using text analysis and heuristics
"""

import os
import re
import pandas as pd
import numpy as np
import math
from collections import Counter

class SmartPricingPredictor:
    def __init__(self):
        # Initialize keyword dictionaries for feature extraction
        self.premium_keywords = [
            'premium', 'luxury', 'deluxe', 'professional', 'high-quality', 'authentic',
            'original', 'certified', 'exclusive', 'superior', 'elite', 'top-quality'
        ]
        
        self.budget_keywords = [
            'budget', 'affordable', 'economy', 'basic', 'standard', 'value',
            'cheap', 'discount', 'sale', 'clearance'
        ]
        
        self.brand_indicators = [
            'brand', 'trademark', 'registered', '®', '™', 'corp', 'inc',
            'company', 'ltd', 'llc', 'official'
        ]
        
        self.material_quality = {
            'gold': 50, 'silver': 20, 'platinum': 80, 'diamond': 100,
            'leather': 15, 'wood': 10, 'metal': 8, 'plastic': 2,
            'cotton': 5, 'silk': 25, 'wool': 15, 'cashmere': 40,
            'stainless steel': 12, 'aluminum': 5, 'titanium': 30
        }
        
        self.category_multipliers = {
            'electronics': 1.5, 'jewelry': 2.0, 'clothing': 1.2, 'home': 1.0,
            'tools': 1.3, 'automotive': 1.4, 'sports': 1.1, 'books': 0.8,
            'toys': 0.9, 'food': 0.7, 'health': 1.2, 'beauty': 1.1
        }
    
    def extract_numeric_features(self, text):
        """Extract numeric values that might indicate price-relevant features"""
        if not isinstance(text, str):
            return {'numbers': [], 'max_number': 0, 'avg_number': 0, 'number_count': 0}
            
        # Find all numbers in the text
        numbers = re.findall(r'\d+\.?\d*', text)
        numbers = [float(x) for x in numbers if float(x) < 10000]  # Filter unreasonable values
        
        if numbers:
            return {
                'numbers': numbers,
                'max_number': max(numbers),
                'avg_number': np.mean(numbers),
                'number_count': len(numbers)
            }
        else:
            return {'numbers': [], 'max_number': 0, 'avg_number': 0, 'number_count': 0}
    
    def extract_quantity_info(self, text):
        """Extract quantity/pack information"""
        if not isinstance(text, str):
            return 1.0
            
        text_lower = text.lower()
        
        # Look for pack quantity patterns
        pack_patterns = [
            r'pack of (\d+)',
            r'(\d+)\s*pack',
            r'(\d+)\s*piece',
            r'(\d+)\s*count',
            r'quantity[:\s]*(\d+)',
            r'(\d+)\s*items?',
            r'set of (\d+)'
        ]
        
        for pattern in pack_patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))
        
        # Look for IPQ (Item Pack Quantity) specific patterns
        ipq_match = re.search(r'value:\s*(\d+\.?\d*)', text_lower)
        if ipq_match:
            return float(ipq_match.group(1))
            
        return 1.0
    
    def analyze_text_sentiment(self, text):
        """Simple sentiment analysis for premium vs budget indicators"""
        if not isinstance(text, str):
            return 0
            
        text_lower = text.lower()
        
        premium_score = sum(1 for keyword in self.premium_keywords if keyword in text_lower)
        budget_score = sum(1 for keyword in self.budget_keywords if keyword in text_lower)
        
        # Return net premium sentiment
        return premium_score - budget_score
    
    def detect_category(self, text):
        """Simple category detection"""
        if not isinstance(text, str):
            return 'home'
            
        text_lower = text.lower()
        
        category_keywords = {
            'electronics': ['electronic', 'device', 'gadget', 'tech', 'digital', 'smart', 'bluetooth', 'wifi'],
            'jewelry': ['jewelry', 'ring', 'necklace', 'bracelet', 'earring', 'watch', 'gold', 'silver'],
            'clothing': ['shirt', 'pants', 'dress', 'jacket', 'clothing', 'apparel', 'fashion', 'wear'],
            'tools': ['tool', 'drill', 'hammer', 'wrench', 'saw', 'equipment', 'hardware'],
            'automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'automotive', 'motorcycle'],
            'sports': ['sport', 'fitness', 'exercise', 'gym', 'outdoor', 'athletic', 'training'],
            'food': ['food', 'snack', 'drink', 'beverage', 'cooking', 'kitchen', 'recipe', 'gourmet'],
            'health': ['health', 'medical', 'vitamin', 'supplement', 'wellness', 'care'],
            'beauty': ['beauty', 'cosmetic', 'makeup', 'skincare', 'fragrance', 'perfume'],
            'books': ['book', 'novel', 'guide', 'manual', 'reading', 'literature'],
            'toys': ['toy', 'game', 'play', 'kids', 'children', 'fun'],
            'home': ['home', 'house', 'decor', 'furniture', 'kitchen', 'bedroom', 'living']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
                
        return 'home'  # Default category
    
    def detect_materials(self, text):
        """Detect premium materials"""
        if not isinstance(text, str):
            return 0
            
        text_lower = text.lower()
        material_score = 0
        
        for material, score in self.material_quality.items():
            if material in text_lower:
                material_score += score
                
        return min(material_score, 100)  # Cap the score
    
    def calculate_text_complexity_score(self, text):
        """Calculate a complexity score based on text features"""
        if not isinstance(text, str):
            return 0
            
        # Basic metrics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text))
        
        # Complexity indicators
        complexity_score = 0
        
        # Length bonus (longer descriptions often indicate more detailed/premium products)
        if word_count > 100:
            complexity_score += 5
        elif word_count > 50:
            complexity_score += 2
            
        # Technical terms bonus
        technical_terms = ['specification', 'feature', 'technology', 'advanced', 'innovative']
        complexity_score += sum(2 for term in technical_terms if term.lower() in text.lower())
        
        return min(complexity_score, 20)  # Cap the bonus
    
    def predict_price(self, sample_id, catalog_content, image_link):
        """Main prediction function"""
        try:
            # Initialize base price
            base_price = 25.0  # Starting point
            
            # Extract various features
            numeric_features = self.extract_numeric_features(catalog_content)
            quantity = self.extract_quantity_info(catalog_content)
            sentiment_score = self.analyze_text_sentiment(catalog_content)
            category = self.detect_category(catalog_content)
            material_score = self.detect_materials(catalog_content)
            complexity_score = self.calculate_text_complexity_score(catalog_content)
            
            # Text length influence (normalized)
            text_length = len(str(catalog_content)) if catalog_content else 0
            length_factor = min(2.0, text_length / 1000)  # Cap at 2x
            
            # Brand presence
            brand_score = sum(1 for indicator in self.brand_indicators 
                            if indicator.lower() in str(catalog_content).lower())
            brand_factor = 1 + (brand_score * 0.1)  # Small boost for brand indicators
            
            # Category multiplier
            category_multiplier = self.category_multipliers.get(category, 1.0)
            
            # Sentiment adjustment
            sentiment_factor = 1 + (sentiment_score * 0.05)  # 5% per premium keyword net
            
            # Material quality bonus
            material_factor = 1 + (material_score * 0.01)  # 1% per material point
            
            # Quantity adjustment (more items = higher total price, but diminishing returns)
            quantity_factor = 1 + (math.log(max(1, quantity)) * 0.2)
            
            # Numeric features influence (could be dimensions, capacity, etc.)
            numeric_factor = 1.0
            if numeric_features['max_number'] > 0:
                # Use log to prevent extreme values
                numeric_factor = 1 + (math.log(max(1, numeric_features['max_number'])) * 0.05)
            
            # Image quality indicator (basic)
            image_factor = 1.0
            if isinstance(image_link, str) and image_link:
                if any(quality in image_link.lower() for quality in ['large', 'high', 'hd']):
                    image_factor = 1.05  # Small boost for higher quality images
            
            # Combine all factors
            final_price = (base_price * 
                          length_factor * 
                          brand_factor * 
                          category_multiplier * 
                          sentiment_factor * 
                          material_factor * 
                          quantity_factor * 
                          numeric_factor * 
                          image_factor)
            
            # Add complexity bonus
            final_price += complexity_score
            
            # Add some randomness to avoid identical predictions
            random_factor = 0.9 + (hash(str(sample_id)) % 21) * 0.01  # 0.9 to 1.1
            final_price *= random_factor
            
            # Ensure reasonable bounds
            final_price = max(1.0, min(500.0, final_price))
            
            return round(final_price, 2)
            
        except Exception as e:
            # Fallback to simple heuristic
            text_len = len(str(catalog_content)) if catalog_content else 100
            fallback_price = max(5.0, min(100.0, text_len / 50))
            return round(fallback_price, 2)

# Global predictor instance
pricing_model = SmartPricingPredictor()

def predictor(sample_id, catalog_content, image_link):
    """
    Enhanced predictor function compatible with the original sample_code.py structure
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    
    Returns:
    - price: Predicted price as a float
    """
    return pricing_model.predict_price(sample_id, catalog_content, image_link)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    print("Smart Product Pricing - Enhanced Heuristic Model")
    print("=" * 55)
    
    # Read test data
    print("Loading test data...")
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    print(f"Loaded {len(test)} test samples")
    
    # Apply predictor function to each row
    print("Generating predictions...")
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']), 
        axis=1
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"\nPrice statistics:")
    print(f"  Mean: ${output_df['price'].mean():.2f}")
    print(f"  Median: ${output_df['price'].median():.2f}")
    print(f"  Min: ${output_df['price'].min():.2f}")
    print(f"  Max: ${output_df['price'].max():.2f}")
    print(f"  Std: ${output_df['price'].std():.2f}")
    
    print(f"\nSample predictions:")
    print(output_df.head(10).to_string(index=False))
    
    # Validate output format
    print(f"\nValidation:")
    print(f"  Output shape: {output_df.shape}")
    print(f"  Required columns present: {list(output_df.columns) == ['sample_id', 'price']}")
    print(f"  All prices positive: {(output_df['price'] > 0).all()}")
    print(f"  No missing values: {not output_df.isnull().any().any()}")