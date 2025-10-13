# Project Documentation

## Overview
This repository contains solutions for the product price prediction challenge. The main focus is on achieving the lowest possible SMAPE score using advanced machine learning and ensemble techniques.

## Directory Structure
- `meta_learning_model.py`: Meta-learning stacking model (best SMAPE so far)
- `ultimate_optimized_model.py`: Advanced ensemble model
- `neural_enhanced_model.py`: Neural network ensemble
- `computer_vision_model.py`: Model with computer vision features
- `less_optimized_models/`: All earlier and less effective models, as well as documentation files
- `dataset/`: Training and test data
- `src/`: Utilities and supporting scripts

## Current Best Models
1. `meta_learning_model.py` (SMAPE: 50.12%)
2. `neural_enhanced_model.py` (SMAPE: 50.45%)
3. `ultimate_optimized_model.py` (SMAPE: 50.97%)
4. `computer_vision_model.py` (SMAPE: 55.30%)

## Next Steps
- Further reduce SMAPE (target: ~48%) via advanced ensembling, feature engineering, or hyperparameter optimization.

## Notes
- All documentation and less optimized models are now in `less_optimized_models/` for a cleaner workspace.
- For details on previous approaches, see the markdown files in `less_optimized_models/`.

## Updated October 2025

- Project folder cleaned and organized for GitHub upload
- All less optimized, ensemble, and experimental models moved to `less_optimized_models/`
- Documentation improved (see README.md and Documentation_template.md)
- .gitignore added for Python, data, and cache files
