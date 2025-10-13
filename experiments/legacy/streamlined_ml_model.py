"""
Streamlined ML Model for <44% SMAPE
Focus on core ML functionality without complex evaluation
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StreamlinedMLModel:
    def __init__(self):
        """Initialize streamlined ML model"""
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def extract_features(self, catalog_content, sample_id=None):
        """Extract optimized features"""
        if pd.isna(catalog_content):
            return self._default_features()
        
        text = str(catalog_content)
        text_lower = text.lower()
        features = {}
        
        # Core features based on correlation analysis
        features['quantity'] = self._extract_quantity(text_lower)
        features['pack_size_log'] = np.log1p(features['quantity'])
        features['is_bulk'] = 1 if features['quantity'] > 1 else 0
        
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_per_word'] = features['text_length'] / max(features['word_count'], 1)
        
        # Number features
        numbers = self._extract_numbers(text)
        features['number_count'] = len(numbers)
        features['max_number'] = max(numbers) if numbers else 0
        features['avg_number'] = np.mean(numbers) if numbers else 0
        
        # Quality indicators
        features['qual_budget'] = self._count_keywords(text_lower, ['budget', 'affordable', 'economy', 'basic'])
        features['qual_premium'] = self._count_keywords(text_lower, ['premium', 'luxury', 'deluxe', 'professional'])
        
        # Category features
        features['cat_automotive'] = self._count_keywords(text_lower, ['car', 'auto', 'vehicle', 'tire', 'engine'])
        features['cat_jewelry'] = self._count_keywords(text_lower, ['jewelry', 'ring', 'necklace', 'watch', 'gold'])
        features['cat_health'] = self._count_keywords(text_lower, ['health', 'vitamin', 'supplement', 'organic'])
        features['cat_electronics'] = self._count_keywords(text_lower, ['electronic', 'device', 'smart', 'digital'])
        features['cat_tools'] = self._count_keywords(text_lower, ['tool', 'drill', 'hammer', 'equipment'])
        features['cat_home'] = self._count_keywords(text_lower, ['home', 'kitchen', 'furniture', 'house'])
        features['cat_food'] = self._count_keywords(text_lower, ['food', 'snack', 'gourmet', 'cooking'])
        
        # Brand features
        features['brand_indicators'] = self._count_keywords(text_lower, ['brand', 'official', 'licensed', 'certified'])
        
        # Size features
        features['size_large'] = self._count_keywords(text_lower, ['large', 'big', 'jumbo', 'xl'])
        features['size_small'] = self._count_keywords(text_lower, ['small', 'mini', 'compact', 'xs'])
        
        # Material features
        features['material_precious'] = self._count_keywords(text_lower, ['gold', 'silver', 'platinum', 'diamond'])
        
        # Derived features
        features['category_total'] = (features['cat_automotive'] + features['cat_jewelry'] + 
                                    features['cat_health'] + features['cat_electronics'] + 
                                    features['cat_tools'] + features['cat_home'] + features['cat_food'])
        
        return features
    
    def _extract_quantity(self, text_lower):
        """Extract quantity"""
        patterns = [
            r'pack of (\d+)', r'(\d+)\s*pack', r'(\d+)\s*count', r'set of (\d+)',
            r'value:\s*(\d+\.?\d*)', r'(\d+)\s*piece', r'(\d+)\s*items?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return min(float(match.group(1)), 50)
        
        return 1.0
    
    def _extract_numbers(self, text):
        """Extract numbers"""
        numbers = re.findall(r'\d+\.?\d*', str(text))
        return [float(x) for x in numbers if 0.1 <= float(x) <= 1000]
    
    def _count_keywords(self, text_lower, keywords):
        """Count keywords"""
        return sum(1 for keyword in keywords if keyword in text_lower)
    
    def _default_features(self):
        """Default features"""
        return {
            'quantity': 1.0, 'pack_size_log': 0.0, 'is_bulk': 0,
            'sentence_count': 1, 'text_length': 0, 'word_count': 0, 'char_per_word': 0,
            'number_count': 0, 'max_number': 0, 'avg_number': 0,
            'qual_budget': 0, 'qual_premium': 0,
            'cat_automotive': 0, 'cat_jewelry': 0, 'cat_health': 0, 'cat_electronics': 0,
            'cat_tools': 0, 'cat_home': 0, 'cat_food': 0,
            'brand_indicators': 0, 'size_large': 0, 'size_small': 0,
            'material_precious': 0, 'category_total': 0
        }
    
    def train(self, train_file, sample_size=20000):
        """Train the models"""
        print(f"Training on {sample_size} samples...")
        
        # Load data
        train_data = pd.read_csv(train_file, nrows=sample_size)
        
        # Extract features
        print("Extracting features...")
        feature_list = []
        for i, row in train_data.iterrows():
            if i % 2000 == 0:
                print(f"  {i}/{len(train_data)}")
            
            features = self.extract_features(row['catalog_content'], row['sample_id'])
            feature_list.append(features)
        
        X = pd.DataFrame(feature_list).fillna(0)
        y = train_data['price'].values
        
        self.feature_names = X.columns.tolist()
        
        print(f"Feature matrix: {X.shape}")
        print(f"Target range: ${y.min():.2f} - ${y.max():.2f}")
        
        # Train models
        print("Training Random Forest...")
        self.rf_model = RandomForestRegressor(
            n_estimators=150, 
            max_depth=12, 
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)
        
        print("Training Gradient Boosting...")
        self.gb_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.15,
            max_depth=6,
            random_state=42
        )
        self.gb_model.fit(X, y)
        
        self.is_trained = True
        print("Training completed!")
        
        return X, y
    
    def predict(self, catalog_content, sample_id):
        """Make prediction"""
        if not self.is_trained:
            return 14.0
        
        try:
            features = self.extract_features(catalog_content, sample_id)
            X = pd.DataFrame([features]).fillna(0)
            X = X.reindex(columns=self.feature_names, fill_value=0)
            
            # Get predictions
            pred_rf = self.rf_model.predict(X)[0]
            pred_gb = self.gb_model.predict(X)[0]
            
            # Ensemble average
            ensemble_pred = (pred_rf * 0.6 + pred_gb * 0.4)
            
            # Bounds
            ensemble_pred = max(0.5, min(300.0, ensemble_pred))
            
            return round(ensemble_pred, 2)
            
        except Exception as e:
            return 14.0

# Global model
ml_model = StreamlinedMLModel()

def predictor(sample_id, catalog_content, image_link):
    """ML predictor function"""
    return ml_model.predict(catalog_content, sample_id)

def main():
    """Main training and evaluation"""
    print("Streamlined ML Model for <44% SMAPE")
    print("=" * 40)
    
    # Train
    X, y = ml_model.train('dataset/train.csv', sample_size=15000)
    
    # Quick validation
    print("\\nQuick validation...")
        # Load validation data from a different part of the training set
        all_train_data = pd.read_csv('dataset/train.csv')
        val_data = all_train_data.iloc[15000:17000].copy()
    
    predictions = []
    actuals = []
    
    for i, row in val_data.iterrows():
        if i % 400 == 0:
            print(f"Validation {i}/2000...")
        
        pred = predictor(row['sample_id'], row['catalog_content'], row['image_link'])
        predictions.append(pred)
        actuals.append(row['price'])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate SMAPE
    smape = np.mean(np.abs(predictions - actuals) / ((np.abs(actuals) + np.abs(predictions)) / 2)) * 100
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    print(f"\\n=== VALIDATION RESULTS ===")
    print(f"SMAPE: {smape:.2f}%")
    print(f"Correlation: {correlation:.4f}")
    print(f"Pred mean: ${predictions.mean():.2f}")
    print(f"Actual mean: ${actuals.mean():.2f}")
    print(f"Pred range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"Actual range: ${actuals.min():.2f} - ${actuals.max():.2f}")
    
    if smape < 60:  # If promising, generate test predictions
        print(f"\\nGenerating test predictions...")
        test = pd.read_csv('dataset/test.csv')
        
        test_predictions = []
        for i, row in test.iterrows():
            if i % 15000 == 0:
                print(f"Test prediction {i}/{len(test)}...")
            
            pred = predictor(row['sample_id'], row['catalog_content'], row['image_link'])
            test_predictions.append(pred)
        
        # Save
        output_df = pd.DataFrame({
            'sample_id': test['sample_id'],
            'price': test_predictions
        })
        
        output_filename = 'dataset/test_out_ml_streamlined.csv'
        output_df.to_csv(output_filename, index=False)
        
        print(f"\\nML predictions saved to {output_filename}")
        print(f"Test stats: Mean ${np.mean(test_predictions):.2f}, Range ${min(test_predictions):.2f}-${max(test_predictions):.2f}")
        
        return smape
    
    return smape

if __name__ == "__main__":
    final_smape = main()
    print(f"\\nFinal SMAPE: {final_smape:.2f}%")
    if final_smape < 44:
        print("🎉 TARGET ACHIEVED!")
    else:
        print("❌ Need more optimization")