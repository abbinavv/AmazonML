# Important Files Not on GitHub

Due to GitHub's file size limitations, some important files are stored locally only:

## 📊 Large Dataset Files (Not on GitHub)
- `dataset/train.csv` (~50MB) - Main training dataset
- `dataset/test.csv` (~25MB) - Test dataset for predictions

**How to get them**: These are the original competition datasets. Download from the source or contact repository owner.

## 🖼️ Downloaded Images (Not on GitHub) 
- `image_cache/*.jpg` (2,183 files, ~500MB total) - Product images for computer vision
- Only `image_cache/image_features.json` is included (metadata)

**How to get them**: Run the download script:
```bash
python scripts/download_computer_vision_images.py
```

## ✅ What IS on GitHub
- ✅ All Python code and models
- ✅ Best model predictions (`results/test_out_meta_learning.csv`)
- ✅ Image metadata (`image_cache/image_features.json`)
- ✅ Sample datasets for testing
- ✅ Complete documentation
- ✅ All requirements and setup files

## 🚀 Quick Setup After Cloning
```bash
git clone https://github.com/abbinavv/AmazonML.git
cd AmazonML
python -m venv .venv
source .venv/bin/activate  # (.venv\Scripts\activate on Windows)
pip install -r requirements.txt

# Optional: Download images for computer vision
python scripts/download_computer_vision_images.py
```

## 📈 Best Results Available
The repository includes the **best model predictions** (50.12% SMAPE) in:
- `results/test_out_meta_learning.csv` - Meta-learning model (best)
- `results/test_out_neural_enhanced.csv` - Neural enhanced model