"""
Example Usage of the Integrated Weather Regulation Prediction System

This script demonstrates various ways to use both the legacy and new modular pipelines.
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configuration imports
# Data pipeline imports
from data.data_loader import DataLoader
from data.data_validation import DataValidator
from data.feature_engineering import AutomatedFeatureEngineer

# Training imports
from run_experiments import ExperimentRunner

# Model imports
from scripts.legacy.model import Dly_Classifier
from src.config import DataConfig, ExperimentConfig, LSTMConfig, TrainingConfig
from src.config_utils import save_config


def example_legacy_pipeline():
    """
    Example 1: Using the legacy pipeline (backward compatible)
    """
    print("\n" + "=" * 60)
    print("Example 1: Legacy Pipeline (Backward Compatible)")
    print("=" * 60)

    # Initialize with legacy approach
    time_init = dt.datetime(2017, 1, 1, 0, 50)
    time_end = dt.datetime(2019, 12, 8, 23, 50)
    data_path = "./Data"
    output_path = "./Output"
    airport = "EGLL"
    time_delta = 30
    download_type = 1

    # Create classifier using legacy mode
    classifier = Dly_Classifier(
        time_init=time_init,
        time_end=time_end,
        info_path=data_path,
        Output_path=output_path,
        AD=airport,
        time_delta=time_delta,
        download_type=download_type,
        use_new_pipeline=False,  # Use legacy pipeline
    )

    # Run legacy experiment
    classifier.run_legacy_experiment()

    print("\nLegacy pipeline completed successfully!")
    print(f"Results saved to: {output_path}")


def example_modular_pipeline():
    """
    Example 2: Using the new modular pipeline with configuration
    """
    print("\n" + "=" * 60)
    print("Example 2: New Modular Pipeline")
    print("=" * 60)

    # Create configuration
    config = ExperimentConfig(
        name="weather_regulation_lstm",
        description="LSTM model for weather-based regulation prediction",
        # Data configuration
        data=DataConfig(
            airports=["EGLL", "LSZH"],
            start_date="2017-01-01",
            end_date="2019-12-08",
            time_step_minutes=30,
            data_path="./Data",
            output_path="./Output",
            feature_engineering=True,
        ),
        # Models to train
        models=["lstm", "transformer", "ensemble"],
        # Training configuration
        training=TrainingConfig(
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=10,
            cross_validation_folds=5,
        ),
        # Model configurations
        model_configs={
            "lstm": LSTMConfig(
                units=[128, 64], dropout_rate=0.3, bidirectional=True, attention=True
            )
        },
    )

    # Save configuration
    save_config(config, "./configs/my_experiment.yaml")

    # Create classifier with configuration
    classifier = Dly_Classifier(
        time_init=dt.datetime(2017, 1, 1),
        time_end=dt.datetime(2019, 12, 8, 23, 59),
        info_path="./Data",
        Output_path="./Output",
        AD="EGLL",
        time_delta=30,
        download_type=1,
        config=config,
        use_new_pipeline=True,
    )

    # Run modular experiment
    classifier.run_modular_experiment(models=["lstm", "transformer"], hyperparameter_tuning=True)

    print("\nModular pipeline completed successfully!")
    print("HTML report generated in Output directory")


def example_data_pipeline():
    """
    Example 3: Using the data pipeline directly
    """
    print("\n" + "=" * 60)
    print("Example 3: Data Pipeline Usage")
    print("=" * 60)

    # Create data configuration
    config = DataConfig(
        airports=["EGLL"], start_date="2017-01-01", end_date="2017-12-31", time_step_minutes=30
    )

    # Initialize data loader
    data_loader = DataLoader(config)

    # Load all data
    print("Loading weather data...")
    data = data_loader.load_all_data(
        airports=["EGLL"], start_date="2017-01-01", end_date="2017-12-31"
    )

    print("\nLoaded data shapes:")
    for name, df in data.items():
        if df is not None and len(df) > 0:
            print(f"  {name}: {df.shape}")

    # Validate data
    print("\nValidating data...")
    validator = DataValidator(config)
    validation_results = validator.validate_dataset(data["features"])

    for name, result in validation_results.items():
        print(f"\n{name} validation:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")

    # Engineer features
    print("\nEngineering features...")
    feature_engineer = AutomatedFeatureEngineer(
        config, max_features=100, selection_method="mutual_info"
    )

    # Create target variable
    y = (
        data["features"]["has_regulation"]
        if "has_regulation" in data["features"].columns
        else np.random.randint(0, 2, len(data["features"]))
    )

    # Apply feature engineering
    features_engineered = feature_engineer.fit_transform(data["features"], y)

    print(f"\nEngineered features shape: {features_engineered.shape}")
    print(f"Selected features: {len(feature_engineer.selected_features)}")


def example_model_comparison():
    """
    Example 4: Compare different models
    """
    print("\n" + "=" * 60)
    print("Example 4: Model Comparison")
    print("=" * 60)

    # Create configuration for comparison
    config = ExperimentConfig(
        name="model_comparison",
        description="Compare different model architectures",
        data=DataConfig(
            airports=["EGLL"], start_date="2017-01-01", end_date="2018-12-31", time_step_minutes=30
        ),
        # Compare multiple models
        models=["random_forest", "lstm", "gru", "transformer", "ensemble"],
        # Disable hyperparameter tuning for fair comparison
        hyperparameter_tuning=False,
        training=TrainingConfig(epochs=50, batch_size=32, test_size=0.2, validation_size=0.2),
    )

    # Run experiment
    runner = ExperimentRunner(config)

    # Use mock data for demonstration
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)

    results = runner.run_experiment(data=(X, y), hyperparameter_tuning=False)

    # Display results
    print("\nModel Comparison Results:")
    print("-" * 40)
    for model_name, metrics in results.items():
        if "test_accuracy" in metrics:
            print(f"{model_name}:")
            print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  F1 Score: {metrics.get('test_f1', 0):.4f}")
            print(f"  Training Time: {metrics.get('training_time', 0):.2f}s")


def example_hyperparameter_tuning():
    """
    Example 5: Hyperparameter tuning
    """
    print("\n" + "=" * 60)
    print("Example 5: Hyperparameter Tuning")
    print("=" * 60)

    # Create configuration with hyperparameter ranges
    config = ExperimentConfig(
        name="hyperparameter_tuning",
        description="Tune LSTM hyperparameters",
        models=["lstm"],
        hyperparameter_tuning=True,
        tuning_method="bayesian",
        n_tuning_trials=20,
        model_configs={
            "lstm": LSTMConfig(
                # Define ranges for tuning
                units=[64, 128, 256],
                layers=[1, 2, 3],
                dropout_rate=[0.2, 0.3, 0.4, 0.5],
                learning_rate=[0.0001, 0.001, 0.01],
                batch_size=[16, 32, 64],
            )
        },
    )

    print("Configuration for hyperparameter tuning:")
    print("  Model: LSTM")
    print(f"  Tuning method: {config.tuning_method}")
    print(f"  Number of trials: {config.n_tuning_trials}")
    print(f"  Units to try: {config.model_configs['lstm'].units}")
    print(f"  Dropout rates: {config.model_configs['lstm'].dropout_rate}")

    # Note: Actual tuning would be run through the experiment runner


def example_command_line_usage():
    """
    Example 6: Command line usage
    """
    print("\n" + "=" * 60)
    print("Example 6: Command Line Usage")
    print("=" * 60)

    print("The model.py script can be run from command line with various options:")
    print("\n1. Run legacy pipeline (default):")
    print("   python model.py")

    print("\n2. Run modular pipeline:")
    print("   python model.py --pipeline modular")

    print("\n3. Run specific models:")
    print("   python model.py --pipeline modular --models lstm transformer")

    print("\n4. Use configuration file:")
    print("   python model.py --config configs/lstm_tuning.yaml")

    print("\n5. Disable hyperparameter tuning:")
    print("   python model.py --pipeline modular --no-tuning")

    print("\n6. Specify date range:")
    print("   python model.py --start-date 2018-01-01 --end-date 2018-12-31")

    print("\n7. Compare pipelines:")
    print("   python model.py --pipeline compare --airport LSZH")


def example_pipeline_comparison():
    """
    Example 7: Compare legacy vs modular pipeline
    """
    print("\n" + "=" * 60)
    print("Example 7: Pipeline Comparison")
    print("=" * 60)

    # This would compare results between pipelines
    print("Comparing pipelines for airport EGLL...")
    print("(This is a demonstration - actual comparison requires data)")

    # Mock results for demonstration
    print("\nLEGACY PIPELINE:")
    print("  FNN Accuracy: 0.8234")
    print("  Training time: 45.3s")

    print("\nMODULAR PIPELINE:")
    print("  FNN Accuracy: 0.8456")
    print("  LSTM Accuracy: 0.8823")
    print("  Transformer Accuracy: 0.8912")
    print("  Ensemble Accuracy: 0.9034")
    print("  Total training time: 234.5s")

    print("\nIMPROVEMENT: +9.7% accuracy with ensemble model")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("WEATHER REGULATION PREDICTION SYSTEM - USAGE EXAMPLES")
    print("=" * 70)

    # Comment/uncomment examples as needed

    # Example 1: Legacy pipeline (for backward compatibility)
    # example_legacy_pipeline()

    # Example 2: New modular pipeline
    # example_modular_pipeline()

    # Example 3: Data pipeline usage
    example_data_pipeline()

    # Example 4: Model comparison
    example_model_comparison()

    # Example 5: Hyperparameter tuning
    example_hyperparameter_tuning()

    # Example 6: Command line usage
    example_command_line_usage()

    # Example 7: Pipeline comparison
    example_pipeline_comparison()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nFor production use, run:")
    print("  python model.py --config configs/production.yaml")
    print("\nFor quick testing, run:")
    print("  python model.py --config configs/quick_test.yaml")


if __name__ == "__main__":
    main()
