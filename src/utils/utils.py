import os
import glob
from pathlib import Path

def list_models(directory='../../models', pattern='*.joblib'):
    """
    Lists all model files in the specified directory matching the given pattern.
    
    Parameters:
    -----------
    directory : str, default='../../models'
        Directory path where models are stored
    pattern : str, default='*.joblib'
        File pattern to match (e.g., '*.joblib', '*.pkl')
        
    Returns:
    --------
    list
        List of tuples (filename without extension, filepath) for each model
    """
    # Convert to Path object for better path handling
    dir_path = Path(directory)
    
    # Ensure the directory exists
    if not dir_path.exists():
        print(f"Directory {directory} does not exist.")
        return []
    
    # Get all files matching the pattern
    model_files = list(dir_path.glob(pattern))
    
    # Create list of tuples (filename without extension, filepath) and sort by filename
    model_tuples = [(path.stem, str(path)) for path in model_files]
    model_tuples.sort()
    
    # Print found models
    if model_tuples:
        print(f"Found {len(model_tuples)} models:")
        for name, _ in model_tuples:
            print(f"  - {name}")
    else:
        print(f"No models found in {directory} matching pattern '{pattern}'")
    
    return model_tuples


if __name__ == "__main__":
    list_models()

