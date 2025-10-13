"""
Project Restructuring Script
Executes full project organization after image download completion
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the new directory structure"""
    directories = [
        "models",
        "experiments/ensemble", 
        "experiments/optimization",
        "experiments/legacy",
        "results",
        "docs", 
        "scripts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def move_main_models():
    """Move main optimized models to models/ folder"""
    main_models = [
        "meta_learning_model.py",
        "neural_enhanced_model.py", 
        "computer_vision_model.py"
    ]
    
    for model in main_models:
        if os.path.exists(model):
            shutil.move(model, f"models/{model}")
            print(f"Moved {model} to models/")

def move_ensemble_models():
    """Move ensemble models to experiments/ensemble/"""
    ensemble_files = [
        "gradient_boosting_ensemble.py",
        "enhanced_gradient_boosting.py", 
        "super_ensemble.py",
        "validate_super_ensemble.py"
    ]
    
    for file in ensemble_files:
        if os.path.exists(file):
            shutil.move(file, f"experiments/ensemble/{file}")
            print(f"Moved {file} to experiments/ensemble/")

def move_optimization_models():
    """Move optimization experiments to experiments/optimization/"""
    optimization_files = [
        "optimized_meta_learning.py",
        "ultimate_optimized_model.py",
        "final_optimized_model.py"
    ]
    
    for file in optimization_files:
        if os.path.exists(file):
            shutil.move(file, f"experiments/optimization/{file}")
            print(f"Moved {file} to experiments/optimization/")

def move_legacy_models():
    """Move legacy models from less_optimized_models/"""
    if os.path.exists("less_optimized_models"):
        # Move all Python files from less_optimized_models to experiments/legacy
        for item in os.listdir("less_optimized_models"):
            item_path = os.path.join("less_optimized_models", item)
            if os.path.isfile(item_path) and item.endswith('.py'):
                shutil.move(item_path, f"experiments/legacy/{item}")
                print(f"Moved {item} to experiments/legacy/")
        
        # Move subdirectories
        for subdir in ["ensemble", "experimental"]:
            subdir_path = os.path.join("less_optimized_models", subdir)
            if os.path.exists(subdir_path):
                for item in os.listdir(subdir_path):
                    if item.endswith('.py'):
                        src = os.path.join(subdir_path, item)
                        dst = f"experiments/legacy/{subdir}_{item}"
                        shutil.move(src, dst)
                        print(f"Moved {item} to experiments/legacy/")

def move_results():
    """Move result files to results/ folder"""
    result_patterns = [
        "test_out_*.csv",
        "*_analysis_results.csv",
        "smape_*.csv",
        "feature_*.csv"
    ]
    
    import glob
    for pattern in result_patterns:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                shutil.move(file, f"results/{file}")
                print(f"Moved {file} to results/")

def move_documentation():
    """Move documentation to docs/ folder"""
    doc_files = [
        "Documentation_template.md",
        "PROJECT_OVERVIEW.md", 
        "PROJECT_STRUCTURE_PLAN.md"
    ]
    
    for doc in doc_files:
        if os.path.exists(doc):
            shutil.move(doc, f"docs/{doc}")
            print(f"Moved {doc} to docs/")

def move_scripts():
    """Move utility scripts to scripts/ folder"""
    script_files = [
        "download_computer_vision_images.py",
        "recreate_image_cache.py"
    ]
    
    for script in script_files:
        if os.path.exists(script):
            shutil.move(script, f"scripts/{script}")
            print(f"Moved {script} to scripts/")

def create_model_readme():
    """Create README for models folder"""
    content = """# Models

This folder contains the main optimized machine learning models for price prediction.

## Best Performing Models

1. **meta_learning_model.py** - Meta-learning stacking model
   - **SMAPE**: 50.12%
   - **Approach**: Advanced stacking with comprehensive features
   - **Status**: Best performer

2. **neural_enhanced_model.py** - Neural network enhanced model  
   - **SMAPE**: 50.45%
   - **Approach**: Deep learning with feature interactions
   - **Status**: Second best

3. **computer_vision_model.py** - Computer vision model
   - **SMAPE**: ~50%
   - **Approach**: Image feature extraction + text features
   - **Status**: Experimental

## Usage

```python
# Run the best model
python meta_learning_model.py
```

See main README.md for detailed instructions.
"""
    
    with open("models/README.md", "w") as f:
        f.write(content)
    print("Created models/README.md")

def create_results_readme():
    """Create README for results folder"""
    content = """# Results

This folder contains model outputs and performance analysis.

## Key Files

- `test_out_meta_learning.csv` - Best model predictions (50.12% SMAPE)
- `test_out_neural_enhanced.csv` - Neural model predictions (50.45% SMAPE)  
- `smape_analysis_results.csv` - Performance comparison across models
- `feature_correlations.csv` - Feature analysis results

## Performance Summary

| Model | SMAPE | Status |
|-------|-------|--------|
| Meta-Learning | 50.12% | Best |
| Neural Enhanced | 50.45% | Good |
| Computer Vision | ~50% | Experimental |
| Ensemble Models | 70-80% | Need improvement |

Target SMAPE: <48%
"""
    
    with open("results/README.md", "w") as f:
        f.write(content)
    print("Created results/README.md")

def update_gitignore():
    """Update .gitignore for proper GitHub upload"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# Data files (too large for GitHub)
dataset/*.csv
!dataset/sample_*.csv

# Images (use Git LFS or ignore)
image_cache/*.jpg
image_cache/*.png
image_cache/*.jpeg
image_cache/*.gif
image_cache/*.webp
!image_cache/image_features.json

# Results (keep some, ignore large ones)
results/*.csv
!results/sample_*.csv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("Updated .gitignore")

def cleanup_empty_folders():
    """Remove empty folders after restructuring"""
    if os.path.exists("less_optimized_models"):
        try:
            # Remove subdirectories if empty
            for subdir in ["ensemble", "experimental"]:
                subdir_path = os.path.join("less_optimized_models", subdir)
                if os.path.exists(subdir_path) and not os.listdir(subdir_path):
                    os.rmdir(subdir_path)
            
            # Remove main directory if empty  
            if not os.listdir("less_optimized_models"):
                os.rmdir("less_optimized_models")
                print("Removed empty less_optimized_models directory")
        except:
            print("Could not remove less_optimized_models (not empty)")

def execute_full_restructuring():
    """Execute complete project restructuring"""
    print("Starting full project restructuring...")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Move files to appropriate locations
    move_main_models()
    move_ensemble_models() 
    move_optimization_models()
    move_legacy_models()
    move_results()
    move_documentation()
    move_scripts()
    
    # Create documentation
    create_model_readme()
    create_results_readme()
    
    # Update configuration files
    update_gitignore()
    
    # Cleanup
    cleanup_empty_folders()
    
    print("=" * 50)
    print("✅ Project restructuring complete!")
    print("✅ Ready for GitHub upload!")
    print("\nNew structure:")
    print("- models/ - Main optimized models")
    print("- experiments/ - Experimental and legacy models") 
    print("- results/ - Model outputs and analysis")
    print("- docs/ - Documentation")
    print("- scripts/ - Utility scripts")
    print("- src/ - Core utilities")

if __name__ == "__main__":
    execute_full_restructuring()