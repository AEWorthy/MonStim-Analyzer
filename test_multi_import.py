#!/usr/bin/env python3
"""
Test script for the multi-experiment import functionality.

This script demonstrates how the new multi-experiment import feature works
and can be used to test the implementation.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_experiment_structure(base_path: Path, exp_name: str, num_datasets: int = 2) -> Path:
    """Create a test experiment structure with CSV files."""
    exp_path = base_path / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_datasets):
        dataset_name = f"240101 TestAnimal{i:02d} Condition{i+1}"
        dataset_path = exp_path / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy CSV file
        csv_content = """time,channel1,channel2,channel3
0.000,0.1,0.2,0.3
0.001,0.2,0.3,0.4
0.002,0.3,0.4,0.5
"""
        csv_file = dataset_path / "test_data.csv"
        csv_file.write_text(csv_content)
        
        print(f"Created test dataset: {dataset_path}")
    
    return exp_path

def demo_multi_import():
    """Demonstrate the multi-import functionality."""
    print("Multi-Experiment Import Demo")
    print("=" * 50)
    
    # Create test directory structure
    test_root = Path("test_experiments")
    test_root.mkdir(exist_ok=True)
    
    # Create multiple test experiments
    experiment_names = ["Experiment_A", "Experiment_B", "Experiment_C"]
    created_experiments = []
    
    for exp_name in experiment_names:
        exp_path = create_test_experiment_structure(test_root, exp_name, num_datasets=2)
        created_experiments.append(exp_path)
        print(f"Created test experiment: {exp_path}")
    
    print(f"\nCreated {len(created_experiments)} test experiments in: {test_root.absolute()}")
    print("\nStructure:")
    for exp_path in created_experiments:
        print(f"  {exp_path.name}/")
        for dataset_path in sorted(exp_path.iterdir()):
            if dataset_path.is_dir():
                print(f"    {dataset_path.name}/")
                for file_path in sorted(dataset_path.iterdir()):
                    print(f"      {file_path.name}")
    
    print("\nTo test the multi-import feature:")
    print("1. Run the MonStim GUI application")
    print("2. Go to File → Import Multiple Experiments")
    print(f"3. Select the root directory: {test_root.absolute()}")
    print("4. Choose which experiments to import")
    print("5. Click 'Import Selected'")
    
    print(f"\nTest data location: {test_root.absolute()}")
    
    return test_root, created_experiments

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the demo
    test_root, experiments = demo_multi_import()
    
    print("\nDemo completed successfully!")
    print("You can now test the multi-experiment import feature in the GUI.")
