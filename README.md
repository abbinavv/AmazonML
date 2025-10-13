# 🏆 AmazonML Price Prediction Project

An advanced machine learning project for predicting product prices using ensemble methods, meta-learning, and computer vision techniques. Achieved **50.12% SMAPE** using sophisticated stacking approaches.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd amazonml-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the best model (Meta-Learning)
cd models
python meta_learning_model.py
```

## 📁 Project Structure

```
├── models/                          # 🎯 Main optimized models (BEST)
│   ├── meta_learning_model.py       # 🥇 Best: 50.12% SMAPE
│   ├── neural_enhanced_model.py     # 🥈 50.45% SMAPE  
│   └── computer_vision_model.py     # 🥉 ~50% SMAPE
│
├── experiments/                     # 🧪 Research & development
│   ├── ensemble/                    # Ensemble approaches
│   ├── optimization/                # Advanced optimizations
│   └── legacy/                      # Earlier experiments
│
├── results/                         # 📊 Model outputs & analysis
├── scripts/                         # 🔧 Utility scripts
├── src/                            # 📚 Core utilities
├── dataset/                        # 💾 Training/test data
└── image_cache/                    # 🖼️ Downloaded images (2,183 files)
```

## 🏆 Model Performance

| Model | SMAPE | Status | Features |
|-------|-------|--------|----------|
| **Meta-Learning** | **50.12%** | ✅ Best | Advanced stacking + comprehensive features |
| Neural Enhanced | 50.45% | 🥈 Second | Deep learning + feature interactions |
| Computer Vision | ~50% | 🧪 Experimental | Image features + text analysis |
| Ensemble Models | 70-80% | ❌ Need work | Various ensemble attempts |

**🎯 Target:** <48% SMAPE | **📈 Best Achievement:** 50.12% SMAPE

## 🔧 Key Features

- **🧠 Meta-Learning**: Advanced stacking with 7 base models
- **🖼️ Computer Vision**: Real image feature extraction (2,183+ images)
- **📝 NLP**: Comprehensive text feature engineering
- **⚡ Ensemble Methods**: Multiple ensemble approaches tested
- **📊 Robust Validation**: Cross-validation with proper SMAPE optimization

## 📋 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- lightgbm, xgboost
- PIL (for image processing)
- See `requirements.txt` for full list

## 🚀 Usage Examples

### Run Best Model
```bash
cd models
python meta_learning_model.py
```

### Run Computer Vision Model
```bash
cd models  
python computer_vision_model.py
```

### Resume Image Downloads
```bash
cd scripts
python download_computer_vision_images.py
```

## 📊 Results & Analysis

- **Training Data**: 75,000 samples
- **Test Data**: 75,000 samples  
- **Image Dataset**: 140,587 unique URLs (2,183+ downloaded)
- **Feature Engineering**: 600+ features per model
- **Cross-Validation**: 5-fold stratified

See `results/` folder for detailed performance analysis.

## 🧪 Experimental Work

The `experiments/` folder contains extensive research:

- **Ensemble Methods**: Gradient boosting combinations
- **Optimization**: Advanced hyperparameter tuning
- **Legacy Models**: Early development iterations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Advanced machine learning techniques
- Meta-learning and stacking approaches
- Computer vision for e-commerce
- Ensemble method research

---

**⭐ Star this repo if it helped you!**
