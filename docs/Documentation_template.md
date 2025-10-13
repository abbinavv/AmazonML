# Documentation Template: AmazonML Price Prediction

## 1. Project Description
This repository contains code and resources for predicting product prices using advanced machine learning, stacking, and meta-learning techniques. The main objective is to minimize SMAPE for robust price estimation.

## 2. Folder Structure
- `meta_learning_model.py` — Main meta-learning model (best performer)
- `neural_enhanced_model.py` — Neural network enhanced model
- `computer_vision_model.py` — Image feature-based model
- `src/` — Utilities and supporting scripts
- `dataset/` — Data files (train/test)
- `less_optimized_models/` — Older, experimental, and ensemble models
    - `ensemble/` — Ensemble and validation scripts
    - `experimental/` — Optimized and experimental models

## 3. Usage Instructions
- Install dependencies: `pip install -r requirements.txt`
- Run the main model: `python meta_learning_model.py`
- For other models, run the corresponding script.

## 4. Results
- Best SMAPE achieved: 50.12% (Meta-Learning Model)
- See `smape_analysis_results.csv` for more details.

## 5. Contribution Guidelines
- Fork the repo and submit pull requests for improvements.

## 6. License
MIT
