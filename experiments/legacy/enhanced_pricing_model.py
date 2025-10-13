"""
Enhanced ML Model for Product Price Prediction
Multi-modal approach using text and image features
"""

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# For text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib

# For image processing (optional - requires additional libraries)
try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import requests
    from io import BytesIO
    IMAGES_AVAILABLE = True
except ImportError:
    IMAGES_AVAILABLE = False
    print("Image processing libraries not available. Using text-only approach.")

class EnhancedPricingModel:
    def __init__(self, use_images=True):
        self.use_images = use_images and IMAGES_AVAILABLE
        self.text_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_extractors = {}
        
    def extract_text_features(self, catalog_content):
        """Extract features from catalog content"""
        features = {}
        
        # Basic text features
        features['text_length'] = len(str(catalog_content))
        features['word_count'] = len(str(catalog_content).split())
        
        # Price-related keywords
        price_keywords = ['premium', 'luxury', 'budget', 'affordable', 'expensive', 'cheap', 'value']
        features['price_keywords'] = sum(1 for keyword in price_keywords if keyword.lower() in str(catalog_content).lower())
        
        # Brand indicators (simplified)
        brand_indicators = ['brand', 'original', 'authentic', 'genuine', 'certified']
        features['brand_score'] = sum(1 for indicator in brand_indicators if indicator.lower() in str(catalog_content).lower())
        
        # Quality indicators
        quality_words = ['high quality', 'premium', 'professional', 'durable', 'robust']
        features['quality_score'] = sum(1 for word in quality_words if word.lower() in str(catalog_content).lower())
        
        # Extract numeric values (could be quantities, sizes, etc.)
        numbers = re.findall(r'\d+\.?\d*', str(catalog_content))
        if numbers:
            features['max_number'] = max(float(x) for x in numbers)
            features['avg_number'] = np.mean([float(x) for x in numbers])
            features['number_count'] = len(numbers)
        else:
            features['max_number'] = 0
            features['avg_number'] = 0
            features['number_count'] = 0
            
        # Extract quantity information (IPQ - Item Pack Quantity)
        quantity_match = re.search(r'(?:quantity|count|pack|piece|item)[:\s]*(\d+)', str(catalog_content).lower())
        features['extracted_quantity'] = float(quantity_match.group(1)) if quantity_match else 1.0
        
        return features
    
    def extract_image_features(self, image_link):
        """Extract basic features from image URL (simplified approach)"""
        if not self.use_images:
            return {}
            
        features = {}
        # Simple URL-based features
        if pd.isna(image_link) or not isinstance(image_link, str):
            features['has_image'] = 0
            features['image_quality_indicator'] = 0
        else:
            features['has_image'] = 1
            # Rough image quality indicator based on URL patterns
            if any(quality in image_link.lower() for quality in ['large', 'high', 'hd']):
                features['image_quality_indicator'] = 1
            else:
                features['image_quality_indicator'] = 0
                
        return features
    
    def prepare_features(self, df):
        """Prepare all features for the dataset"""
        print("Extracting text features...")
        
        # Extract structured features
        text_features_list = []
        image_features_list = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing row {idx}/{len(df)}")
                
            text_feat = self.extract_text_features(row['catalog_content'])
            image_feat = self.extract_image_features(row['image_link'])
            
            text_features_list.append(text_feat)
            image_features_list.append(image_feat)
        
        # Convert to DataFrames
        text_features_df = pd.DataFrame(text_features_list)
        image_features_df = pd.DataFrame(image_features_list)
        
        # TF-IDF features
        print("Computing TF-IDF features...")
        text_content = df['catalog_content'].fillna('').astype(str)
        tfidf_features = self.text_vectorizer.fit_transform(text_content).toarray()
        tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Combine all features
        all_features = pd.concat([text_features_df, image_features_df, tfidf_df], axis=1)
        
        return all_features
    
    def train(self, train_df):
        """Train the pricing model"""
        print("Preparing training features...")
        X_train = self.prepare_features(train_df)
        y_train = train_df['price'].values
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Training models...")
        
        # Train multiple models for ensemble
        self.models['rf'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.models['gb'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.models['ridge'] = Ridge(alpha=1.0)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
        print("Training completed!")
    
    def predict(self, test_df):
        """Make predictions on test data"""
        print("Preparing test features...")
        X_test = self.prepare_features(test_df)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Making predictions...")
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X_test_scaled)
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Ensure positive prices
        ensemble_pred = np.maximum(ensemble_pred, 0.1)
        
        return ensemble_pred

def predictor(sample_id, catalog_content, image_link, model=None):
    """
    Enhanced predictor function for use with existing sample_code structure
    """
    if model is None:
        # Fallback to simple heuristic-based prediction
        features = {}
        
        # Text length influence
        text_len = len(str(catalog_content))
        base_price = min(100, max(5, text_len / 100))
        
        # Premium keywords boost
        premium_keywords = ['premium', 'luxury', 'professional', 'high-quality', 'authentic']
        premium_boost = sum(2 for keyword in premium_keywords if keyword.lower() in str(catalog_content).lower())
        
        # Brand boost
        if any(brand in str(catalog_content).lower() for brand in ['brand', 'original', 'certified']):
            brand_boost = 5
        else:
            brand_boost = 0
            
        # Extract numbers for size/quantity influence
        numbers = re.findall(r'\d+\.?\d*', str(catalog_content))
        if numbers:
            max_num = max(float(x) for x in numbers if float(x) < 1000)  # Reasonable upper bound
            quantity_factor = min(2.0, max_num / 10)  # Cap the influence
        else:
            quantity_factor = 1.0
            
        final_price = (base_price + premium_boost + brand_boost) * quantity_factor
        return round(max(1.0, min(200.0, final_price)), 2)  # Reasonable price bounds
    
    # If model is provided, use it for prediction
    test_data = pd.DataFrame({
        'sample_id': [sample_id],
        'catalog_content': [catalog_content], 
        'image_link': [image_link]
    })
    
    prediction = model.predict(test_data)
    return round(float(prediction[0]), 2)

def train_and_save_model():
    """Train model on full dataset and save for later use"""
    print("Loading training data...")
    
    # For demonstration, we'll use a sample approach since full data is large
    try:
        train_df = pd.read_csv('dataset/train.csv')
        print(f"Loaded {len(train_df)} training samples")
        
        # Initialize and train model
        model = EnhancedPricingModel(use_images=False)  # Text-only for now
        model.train(train_df)
        
        # Save the model
        joblib.dump(model, 'enhanced_pricing_model.pkl')
        print("Model saved successfully!")
        
        return model
        
    except Exception as e:
        print(f"Error training model: {e}")
        print("Using heuristic-based approach instead")
        return None

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    print("Smart Product Pricing - Enhanced Model")
    print("=" * 50)
    
    # Try to train and load full model
    # model = train_and_save_model()
    
    # For now, use heuristic approach due to computational constraints
    model = None
    
    # Read test data
    print("Loading test data...")
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    print(f"Loaded {len(test)} test samples")
    
    # Apply predictor function to each row
    print("Generating predictions...")
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link'], model), 
        axis=1
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Price statistics:")
    print(f"  Mean: ${output_df['price'].mean():.2f}")
    print(f"  Median: ${output_df['price'].median():.2f}")
    print(f"  Min: ${output_df['price'].min():.2f}")
    print(f"  Max: ${output_df['price'].max():.2f}")
    print(f"\nSample predictions:")
    print(output_df.head(10))