"""
Deep Analysis of SMAPE Issues
Identifying patterns to reduce SMAPE from 78% to <44%
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

def analyze_smape_issues():
    """Analyze what's causing high SMAPE in current model"""
    print("=== DEEP SMAPE ANALYSIS ===")
    
    # Load larger training sample
    print("Loading training data...")
    train_data = pd.read_csv('dataset/train.csv', nrows=10000)
    
    # Current model predictions on training data
    from sample_code import predictor
    
    predictions = []
    actuals = []
    
    print("Generating predictions for analysis...")
    for i, row in train_data.iterrows():
        if i % 1000 == 0:
            print(f"Processing {i}/10000...")
        
        pred = predictor(row['sample_id'], row['catalog_content'], row['image_link'])
        predictions.append(pred)
        actuals.append(row['price'])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate SMAPE and analyze errors
    smape_values = np.abs(predictions - actuals) / ((np.abs(actuals) + np.abs(predictions)) / 2) * 100
    overall_smape = np.mean(smape_values)
    
    print(f"\n=== SMAPE BREAKDOWN ===")
    print(f"Overall SMAPE: {overall_smape:.2f}%")
    print(f"Median SMAPE: {np.median(smape_values):.2f}%")
    print(f"75th percentile SMAPE: {np.percentile(smape_values, 75):.2f}%")
    print(f"95th percentile SMAPE: {np.percentile(smape_values, 95):.2f}%")
    
    # Identify high error cases
    high_error_indices = np.where(smape_values > 100)[0]
    print(f"\nSamples with SMAPE > 100%: {len(high_error_indices)} ({len(high_error_indices)/len(smape_values)*100:.1f}%)")
    
    if len(high_error_indices) > 0:
        print("\nAnalyzing high-error cases...")
        high_error_actual = actuals[high_error_indices]
        high_error_pred = predictions[high_error_indices]
        
        print(f"High error - Actual price range: ${high_error_actual.min():.2f} - ${high_error_actual.max():.2f}")
        print(f"High error - Predicted price range: ${high_error_pred.min():.2f} - ${high_error_pred.max():.2f}")
        print(f"High error - Actual mean: ${high_error_actual.mean():.2f}")
        print(f"High error - Predicted mean: ${high_error_pred.mean():.2f}")
    
    # Analyze by price ranges
    print(f"\n=== SMAPE BY PRICE RANGE ===")
    price_bins = [0, 5, 10, 20, 50, 100, float('inf')]
    price_labels = ['$0-5', '$5-10', '$10-20', '$20-50', '$50-100', '$100+']
    
    for i, (low, high) in enumerate(zip(price_bins[:-1], price_bins[1:])):
        mask = (actuals >= low) & (actuals < high)
        if np.sum(mask) > 0:
            bin_smape = np.mean(smape_values[mask])
            bin_count = np.sum(mask)
            bin_actual_mean = np.mean(actuals[mask])
            bin_pred_mean = np.mean(predictions[mask])
            print(f"{price_labels[i]:8s}: SMAPE {bin_smape:6.2f}%, Count {bin_count:4d}, Actual ${bin_actual_mean:6.2f}, Pred ${bin_pred_mean:6.2f}")
    
    # Correlation analysis
    print(f"\n=== PREDICTION QUALITY ===")
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    print(f"Prediction-Actual Correlation: {correlation:.4f}")
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    
    # Bias analysis
    bias = np.mean(predictions - actuals)
    print(f"Bias (pred - actual): ${bias:.2f}")
    
    return train_data, predictions, actuals, smape_values

def analyze_feature_patterns(train_data, predictions, actuals):
    """Analyze what features correlate with pricing"""
    print(f"\n=== FEATURE PATTERN ANALYSIS ===")
    
    # Add derived features
    train_data['text_length'] = train_data['catalog_content'].str.len()
    train_data['word_count'] = train_data['catalog_content'].str.split().str.len()
    train_data['prediction'] = predictions
    train_data['actual_price'] = actuals
    train_data['error'] = np.abs(predictions - actuals)
    
    # Text features correlation with price
    print("Text feature correlations with actual price:")
    text_features = ['text_length', 'word_count']
    for feature in text_features:
        corr = train_data[feature].corr(train_data['actual_price'])
        print(f"  {feature}: {corr:.4f}")
    
    # Extract more features
    print("\nExtracting advanced features...")
    
    # Price-relevant keywords
    keywords = {
        'premium': ['premium', 'luxury', 'deluxe', 'professional'],
        'materials': ['gold', 'silver', 'platinum', 'diamond', 'leather'],
        'electronics': ['electronic', 'digital', 'smart', 'bluetooth'],
        'automotive': ['car', 'auto', 'vehicle', 'tire'],
        'jewelry': ['jewelry', 'ring', 'necklace', 'watch'],
        'tools': ['tool', 'drill', 'equipment'],
        'bulk': ['pack', 'set', 'bulk', 'dozen']
    }
    
    for category, words in keywords.items():
        train_data[f'{category}_count'] = train_data['catalog_content'].str.lower().apply(
            lambda x: sum(1 for word in words if word in str(x))
        )
    
    # Analyze keyword correlations
    print("\nKeyword correlations with actual price:")
    for category in keywords.keys():
        col = f'{category}_count'
        if col in train_data.columns:
            corr = train_data[col].corr(train_data['actual_price'])
            print(f"  {category}: {corr:.4f}")
    
    # Find products with very low/high prices
    print(f"\n=== EXTREME PRICE ANALYSIS ===")
    low_price = train_data[train_data['actual_price'] <= 2.0]
    high_price = train_data[train_data['actual_price'] >= 100.0]
    
    print(f"Very low price products (<= $2): {len(low_price)}")
    if len(low_price) > 0:
        print(f"  Avg text length: {low_price['text_length'].mean():.0f}")
        print(f"  Avg prediction: ${low_price['prediction'].mean():.2f}")
    
    print(f"Very high price products (>= $100): {len(high_price)}")
    if len(high_price) > 0:
        print(f"  Avg text length: {high_price['text_length'].mean():.0f}")
        print(f"  Avg prediction: ${high_price['prediction'].mean():.2f}")
    
    return train_data

def identify_improvement_opportunities(train_data):
    """Identify specific areas for model improvement"""
    print(f"\n=== IMPROVEMENT OPPORTUNITIES ===")
    
    # Find systematic biases
    train_data['price_ratio'] = train_data['prediction'] / train_data['actual_price']
    
    # Cases where we severely underpredict
    underpredict = train_data[train_data['price_ratio'] < 0.5]
    print(f"Severe underprediction cases (pred < 0.5 * actual): {len(underpredict)}")
    if len(underpredict) > 0:
        print(f"  Avg actual price: ${underpredict['actual_price'].mean():.2f}")
        print(f"  Avg predicted price: ${underpredict['prediction'].mean():.2f}")
        print(f"  Avg text length: {underpredict['text_length'].mean():.0f}")
    
    # Cases where we severely overpredict
    overpredict = train_data[train_data['price_ratio'] > 2.0]
    print(f"Severe overprediction cases (pred > 2 * actual): {len(overpredict)}")
    if len(overpredict) > 0:
        print(f"  Avg actual price: ${overpredict['actual_price'].mean():.2f}")
        print(f"  Avg predicted price: ${overpredict['prediction'].mean():.2f}")
        print(f"  Avg text length: {overpredict['text_length'].mean():.0f}")
    
    # Extract numeric values from text
    print(f"\nAnalyzing numeric patterns...")
    
    def extract_numbers(text):
        if pd.isna(text):
            return []
        numbers = re.findall(r'\d+\.?\d*', str(text))
        return [float(x) for x in numbers if 0.1 <= float(x) <= 1000]
    
    train_data['numbers'] = train_data['catalog_content'].apply(extract_numbers)
    train_data['max_number'] = train_data['numbers'].apply(lambda x: max(x) if x else 0)
    train_data['number_count'] = train_data['numbers'].apply(len)
    
    # Correlations with numbers
    print(f"Max number correlation with price: {train_data['max_number'].corr(train_data['actual_price']):.4f}")
    print(f"Number count correlation with price: {train_data['number_count'].corr(train_data['actual_price']):.4f}")
    
    return train_data

if __name__ == "__main__":
    # Run comprehensive analysis
    train_data, predictions, actuals, smape_values = analyze_smape_issues()
    train_data = analyze_feature_patterns(train_data, predictions, actuals)
    train_data = identify_improvement_opportunities(train_data)
    
    print(f"\n=== SUMMARY ===")
    print(f"Current SMAPE: {np.mean(smape_values):.2f}%")
    print(f"Target SMAPE: <44%")
    print(f"Improvement needed: {np.mean(smape_values) - 44:.2f} percentage points")
    
    # Save analysis results
    analysis_file = 'smape_analysis_results.csv'
    train_data.to_csv(analysis_file, index=False)
    print(f"\nAnalysis results saved to {analysis_file}")