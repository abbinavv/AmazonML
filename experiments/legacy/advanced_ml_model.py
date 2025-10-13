"""
Advanced ML Model for <44% SMAPE
Using comprehensive features and proper regression techniques
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLPricingModel:
    def __init__(self):
        """Initialize with multiple models for ensemble"""
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        
    def extract_features_optimized(self, catalog_content, sample_id=None):
        """Extract optimized features based on correlation analysis"""
        if pd.isna(catalog_content):
            return self._get_default_features()
        
        text = str(catalog_content)
        text_lower = text.lower()
        features = {}
        
        # Top correlated features from analysis
        
        # 1. Quantity (correlation: 0.1841) - Most important
        features['quantity'] = self._extract_quantity(text_lower)
        features['pack_size_log'] = np.log1p(features['quantity'])
        features['is_bulk'] = 1 if features['quantity'] > 1 else 0
        
        # 2. Sentence count (correlation: 0.1784)
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # 3. Text metrics (correlations: 0.14-0.15)
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_per_word'] = features['text_length'] / max(features['word_count'], 1)
        
        # 4. Number features (correlation: 0.1443)
        numbers = self._extract_numbers(text)
        features['number_count'] = len(numbers)
        features['max_number'] = max(numbers) if numbers else 0
        features['avg_number'] = np.mean(numbers) if numbers else 0
        features['number_range'] = max(numbers) - min(numbers) if len(numbers) > 1 else 0
        
        # 5. Quality indicators (correlations: 0.10-0.14)
        features['qual_budget'] = self._count_keywords(text_lower, ['budget', 'affordable', 'economy', 'basic', 'standard', 'entry-level'])
        features['qual_premium'] = self._count_keywords(text_lower, ['premium', 'luxury', 'deluxe', 'high-end', 'top-quality', 'superior'])
        features['qual_professional'] = self._count_keywords(text_lower, ['professional', 'commercial', 'industrial', 'heavy-duty', 'pro-grade'])
        
        # 6. Category features (correlations: 0.10-0.11)
        features['cat_automotive'] = self._count_keywords(text_lower, ['car', 'auto', 'vehicle', 'tire', 'engine', 'brake', 'oil', 'filter'])
        features['cat_jewelry'] = self._count_keywords(text_lower, ['jewelry', 'ring', 'necklace', 'bracelet', 'earring', 'watch', 'pendant'])
        features['cat_health'] = self._count_keywords(text_lower, ['health', 'vitamin', 'supplement', 'medical', 'wellness', 'organic'])
        features['cat_electronics'] = self._count_keywords(text_lower, ['electronic', 'device', 'gadget', 'smart', 'bluetooth', 'digital'])
        features['cat_tools'] = self._count_keywords(text_lower, ['tool', 'drill', 'hammer', 'saw', 'wrench', 'equipment', 'hardware'])
        features['cat_home'] = self._count_keywords(text_lower, ['home', 'kitchen', 'furniture', 'decor', 'house', 'appliance'])
        features['cat_clothing'] = self._count_keywords(text_lower, ['clothing', 'shirt', 'dress', 'pants', 'jacket', 'fashion', 'apparel'])
        features['cat_food'] = self._count_keywords(text_lower, ['food', 'snack', 'drink', 'beverage', 'cooking', 'gourmet'])
        
        # 7. Brand/exclusivity features
        features['brand_licensed'] = self._count_keywords(text_lower, ['licensed', 'authorized', 'official', 'certified'])
        features['brand_branded'] = self._count_keywords(text_lower, ['brand', 'company', 'corp', 'inc', 'ltd'])
        features['price_exclusive'] = self._count_keywords(text_lower, ['exclusive', 'limited', 'rare', 'special', 'unique'])
        
        # 8. Size features
        features['size_large'] = self._count_keywords(text_lower, ['large', 'big', 'jumbo', 'oversized', 'xl', 'extra large'])
        features['size_small'] = self._count_keywords(text_lower, ['small', 'mini', 'compact', 'tiny', 'xs', 'petite'])
        features['size_medium'] = self._count_keywords(text_lower, ['medium', 'standard', 'regular', 'normal'])
        
        # 9. Material features
        features['material_precious'] = self._count_keywords(text_lower, ['gold', 'silver', 'platinum', 'diamond'])
        features['material_quality'] = self._count_keywords(text_lower, ['leather', 'steel', 'titanium', 'ceramic'])
        
        # 10. Additional derived features
        features['text_complexity'] = features['sentence_count'] / max(features['word_count'], 1) * 100
        features['category_total'] = (features['cat_automotive'] + features['cat_jewelry'] + 
                                    features['cat_health'] + features['cat_electronics'] + 
                                    features['cat_tools'] + features['cat_home'] + 
                                    features['cat_clothing'] + features['cat_food'])
        features['quality_total'] = features['qual_budget'] + features['qual_premium'] + features['qual_professional']
        
        # Sample-specific features for variation
        if sample_id:
            features['sample_hash'] = hash(str(sample_id)) % 1000
            
        return features
    
    def _extract_quantity(self, text_lower):
        """Extract quantity with all patterns"""
        patterns = [
            r'pack of (\d+)', r'(\d+)\s*pack\b', r'(\d+)\s*piece', r'(\d+)\s*count\b',
            r'set of (\d+)', r'(\d+)\s*items?\b', r'quantity[:\s]*(\d+)',
            r'value:\s*(\d+\.?\d*)', r'(\d+)\s*per\s*pack', r'(\d+)\s*units?\b',
            r'(\d+)\s*pair', r'(\d+)\s*dozen', r'(\d+)x\s*pack', r'multipack\s*(\d+)',
            r'bulk\s*(\d+)', r'(\d+)\s*box', r'case\s*of\s*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return min(float(match.group(1)), 100)
        
        if any(word in text_lower for word in ['bulk', 'wholesale', 'case', 'dozen']):
            return 12.0
        
        return 1.0
    
    def _extract_numbers(self, text):
        """Extract relevant numbers from text"""
        if pd.isna(text):
            return []
        
        # Find all numbers
        numbers = re.findall(r'\d+\.?\d*', str(text))
        valid_numbers = []
        
        for num_str in numbers:
            try:
                num = float(num_str)
                if 0.1 <= num <= 10000:  # Reasonable range
                    valid_numbers.append(num)
            except:
                continue
        
        return valid_numbers
    
    def _count_keywords(self, text_lower, keywords):
        """Count keyword occurrences"""
        return sum(1 for keyword in keywords if keyword in text_lower)
    
    def _get_default_features(self):
        """Default features for missing text"""
        return {
            'quantity': 1.0, 'pack_size_log': 0.0, 'is_bulk': 0,
            'sentence_count': 1, 'text_length': 0, 'word_count': 0, 'char_per_word': 0,
            'number_count': 0, 'max_number': 0, 'avg_number': 0, 'number_range': 0,
            'qual_budget': 0, 'qual_premium': 0, 'qual_professional': 0,
            'cat_automotive': 0, 'cat_jewelry': 0, 'cat_health': 0, 'cat_electronics': 0,
            'cat_tools': 0, 'cat_home': 0, 'cat_clothing': 0, 'cat_food': 0,
            'brand_licensed': 0, 'brand_branded': 0, 'price_exclusive': 0,
            'size_large': 0, 'size_small': 0, 'size_medium': 0,
            'material_precious': 0, 'material_quality': 0,
            'text_complexity': 0, 'category_total': 0, 'quality_total': 0,
            'sample_hash': 0
        }
    
    def train_models(self, train_file, sample_size=30000):
        """Train ensemble of models"""
        print(f"Training ML models on {sample_size} samples...")
        
        # Load training data
        train_data = pd.read_csv(train_file, nrows=sample_size)
        
        # Extract features
        print("Extracting features...")
        feature_list = []
        for i, row in train_data.iterrows():
            if i % 5000 == 0:
                print(f"  Processing {i}/{len(train_data)}...")
            
            features = self.extract_features_optimized(row['catalog_content'], row['sample_id'])
            feature_list.append(features)
        
        # Convert to DataFrame
        X = pd.DataFrame(feature_list)
        y = train_data['price'].values
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target range: ${y.min():.2f} - ${y.max():.2f}")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        print("Scaling features...")
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_standard = self.scalers['standard'].fit_transform(X)
        X_robust = self.scalers['robust'].fit_transform(X)
        
        # Train multiple models
        print("Training models...")
        
        # 1. Random Forest (handles non-linear relationships)
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X, y)
        
        # 2. Gradient Boosting (sequential learning)
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
        self.models['gb'].fit(X, y)
        
        # 3. Ridge Regression with standard scaling
        self.models['ridge_std'] = Ridge(alpha=1.0)
        self.models['ridge_std'].fit(X_standard, y)
        
        # 4. Ridge Regression with robust scaling
        self.models['ridge_rob'] = Ridge(alpha=1.0)
        self.models['ridge_rob'].fit(X_robust, y)
        
        # 5. Linear Regression with feature selection
        # Select top features based on RF importance
        rf_importance = self.models['rf'].feature_importances_
        top_features_idx = np.argsort(rf_importance)[-15:]  # Top 15 features
        X_selected = X.iloc[:, top_features_idx]
        
        self.models['linear_selected'] = LinearRegression()
        self.models['linear_selected'].fit(X_selected, y)
        self.top_features_idx = top_features_idx
        
        self.is_trained = True
        
        # Evaluate models
        print("\\nEvaluating models...")
        self._evaluate_models(X, y, X_standard, X_robust)
        
        return X, y
    
    def _evaluate_models(self, X, y, X_standard, X_robust):
        """Evaluate model performance"""
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            if name == 'ridge_std':
                X_eval = X_standard
            elif name == 'ridge_rob':
                X_eval = X_robust
            elif name == 'linear_selected':
                X_eval = X.iloc[:, self.top_features_idx].values
            else:
                X_eval = X.values
            
            # Cross-validation SMAPE
            smape_scores = []
            for train_idx, val_idx in kfold.split(X_eval):
                X_train, X_val = X_eval[train_idx], X_eval[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create a copy of the model for training
                if name == 'rf':
                    temp_model = RandomForestRegressor(
                        n_estimators=200, max_depth=15, min_samples_split=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1
                    )
                elif name == 'gb':
                    temp_model = GradientBoostingRegressor(
                        n_estimators=200, learning_rate=0.1, max_depth=8,
                        subsample=0.8, random_state=42
                    )
                elif name == 'ridge_std':
                    temp_model = Ridge(alpha=1.0)
                elif name == 'ridge_rob':
                    temp_model = Ridge(alpha=1.0)
                elif name == 'linear_selected':
                    temp_model = LinearRegression()
                
                temp_model.fit(X_train, y_train)
                pred = temp_model.predict(X_val)
                
                smape = np.mean(np.abs(pred - y_val) / ((np.abs(y_val) + np.abs(pred)) / 2)) * 100
                smape_scores.append(smape)
            
            avg_smape = np.mean(smape_scores)
            std_smape = np.std(smape_scores)
            print(f"{name:15s}: SMAPE {avg_smape:6.2f}% ± {std_smape:5.2f}%")
    
    def predict_ensemble(self, catalog_content, sample_id):
        """Ensemble prediction from all models"""
        if not self.is_trained:
            return 14.0  # Fallback
        
        try:
            # Extract features
            features = self.extract_features_optimized(catalog_content, sample_id)
            X = pd.DataFrame([features])
            X = X.fillna(0)
            
            # Ensure feature order matches training
            X = X.reindex(columns=self.feature_names, fill_value=0)
            
            # Get predictions from all models
            predictions = []
            
            # Random Forest
            pred_rf = self.models['rf'].predict(X)[0]
            predictions.append(pred_rf)
            
            # Gradient Boosting
            pred_gb = self.models['gb'].predict(X)[0]
            predictions.append(pred_gb)
            
            # Ridge with standard scaling
            X_std = self.scalers['standard'].transform(X)
            pred_ridge_std = self.models['ridge_std'].predict(X_std)[0]
            predictions.append(pred_ridge_std)
            
            # Ridge with robust scaling
            X_rob = self.scalers['robust'].transform(X)
            pred_ridge_rob = self.models['ridge_rob'].predict(X_rob)[0]
            predictions.append(pred_ridge_rob)
            
            # Linear with selected features
            X_sel = X.iloc[:, self.top_features_idx]
            pred_linear = self.models['linear_selected'].predict(X_sel)[0]
            predictions.append(pred_linear)
            
            # Ensemble: weighted average (RF and GB get higher weight)
            weights = [0.3, 0.3, 0.15, 0.15, 0.1]  # RF, GB, Ridge_std, Ridge_rob, Linear
            ensemble_pred = np.average(predictions, weights=weights)
            
            # Ensure positive and reasonable bounds
            ensemble_pred = max(0.1, min(500.0, ensemble_pred))
            
            return round(ensemble_pred, 2)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 14.0

# Global model instance
ml_model = AdvancedMLPricingModel()

def predictor(sample_id, catalog_content, image_link):
    """ML-based predictor for <44% SMAPE target"""
    return ml_model.predict_ensemble(catalog_content, sample_id)

def main():
    """Train and evaluate the advanced ML model"""
    print("Advanced ML Pricing Model for <44% SMAPE")
    print("=" * 50)
    
    # Train models
    X, y = ml_model.train_models('dataset/train.csv', sample_size=25000)
    
    # Test on validation set
    print("\\nTesting on validation set...")
    val_data = pd.read_csv('dataset/train.csv', skiprows=25000, nrows=3000)
    
        # Reset index to ensure we have proper row indexing
        val_data = val_data.reset_index(drop=True)
    
    predictions = []
    actuals = []
    
    for i, row in val_data.iterrows():
        if i % 500 == 0:
            print(f"Validating {i}/3000...")
        
        pred = predictor(row['sample_id'], row['catalog_content'], row['image_link'])
        predictions.append(pred)
        actuals.append(row['price'])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    smape = np.mean(np.abs(predictions - actuals) / ((np.abs(actuals) + np.abs(predictions)) / 2)) * 100
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    print(f"\\n=== VALIDATION RESULTS ===")
    print(f"SMAPE: {smape:.4f}%")
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Prediction mean: ${predictions.mean():.2f}")
    print(f"Actual mean: ${actuals.mean():.2f}")
    print(f"Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}")
    print(f"Actual range: ${actuals.min():.2f} - ${actuals.max():.2f}")
    
    if smape < 44:
        print(f"\\n🎉 TARGET ACHIEVED! SMAPE {smape:.2f}% < 44%")
        print("Generating test predictions...")
        
        # Generate test predictions
        test = pd.read_csv('dataset/test.csv')
        test_predictions = []
        
        for i, row in test.iterrows():
            if i % 10000 == 0:
                print(f"Processing {i}/{len(test)}...")
            
            pred = predictor(row['sample_id'], row['catalog_content'], row['image_link'])
            test_predictions.append(pred)
        
        # Save results
        output_df = pd.DataFrame({
            'sample_id': test['sample_id'],
            'price': test_predictions
        })
        
        output_filename = 'dataset/test_out_ml_optimized.csv'
        output_df.to_csv(output_filename, index=False)
        
        print(f"\\nML predictions saved to {output_filename}")
        print(f"Test prediction statistics:")
        print(f"  Mean: ${output_df['price'].mean():.2f}")
        print(f"  Median: ${output_df['price'].median():.2f}")
        print(f"  Range: ${output_df['price'].min():.2f} - ${output_df['price'].max():.2f}")
        
        return True
    else:
        print(f"\\n❌ Target not achieved. SMAPE {smape:.2f}% >= 44%")
        print("Need further optimization...")
        return False

if __name__ == "__main__":
    success = main()