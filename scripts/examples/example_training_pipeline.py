"""
Test script for the complete training pipeline

This script demonstrates:
1. Using the unified trainer
2. Hyperparameter tuning with different methods
3. Running experiment suites
4. Creating ensembles from results
5. Generating reports
"""

import warnings

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our modules
from config import ModelType
from config_parser import ConfigParser
from config_utils import optimize_config_for_hardware
from run_experiments import ExperimentRunner, ExperimentSuite
from training import create_trainer, create_tuner

warnings.filterwarnings("ignore")


def test_unified_trainer():
    """Test the unified training interface"""
    print("\n" + "=" * 60)
    print("Testing Unified Trainer")
    print("=" * 60)

    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create configuration
    parser = ConfigParser()
    config = parser.create_default_config()
    config.experiment_name = "test_unified_trainer"
    config.model_type = ModelType.RANDOM_FOREST
    config.output_dir = "./test_output/trainer"

    # Create trainer
    trainer = create_trainer(config, enable_tracking=True)

    # Create and train model
    from models.random_forest import RandomForestModel

    model = RandomForestModel(config)

    # Train
    history = trainer.train(
        model=model.model, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test
    )

    print("Training completed!")
    print(f"Validation metrics: {history.get('val_metrics', {})}")

    # Test cross-validation
    print("\nTesting cross-validation...")
    cv_results = trainer.cross_validate(
        model_class=lambda: RandomForestModel(config).model, X=X_train, y=y_train, cv_folds=5
    )

    print("Cross-validation results:")
    print(
        f"Mean accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})"
    )
    print(f"Mean F1: {cv_results['mean_f1']:.4f} (+/- {cv_results['std_f1']:.4f})")


def test_hyperparameter_tuning():
    """Test different hyperparameter tuning methods"""
    print("\n" + "=" * 60)
    print("Testing Hyperparameter Tuning")
    print("=" * 60)

    # Generate sample data
    X, y = make_classification(
        n_samples=500,  # Smaller dataset for faster testing
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    # Test Grid Search
    print("\n1. Testing Grid Search...")
    param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10], "min_samples_split": [2, 5]}

    from sklearn.ensemble import RandomForestClassifier

    grid_tuner = create_tuner(
        method="grid",
        model_class=RandomForestClassifier,
        param_space=param_grid,
        scoring="f1",
        cv=3,
    )

    grid_result = grid_tuner.tune(X, y)
    print(f"Best parameters: {grid_result.best_params}")
    print(f"Best score: {grid_result.best_score:.4f}")

    # Test Random Search
    print("\n2. Testing Random Search...")
    param_distributions = {
        "n_estimators": {"type": "int", "low": 50, "high": 200},
        "max_depth": {"type": "int", "low": 5, "high": 20},
        "min_samples_split": {"type": "int", "low": 2, "high": 10},
    }

    random_tuner = create_tuner(
        method="random",
        model_class=RandomForestClassifier,
        param_space=param_distributions,
        n_iter=10,
        scoring="f1",
        cv=3,
    )

    random_result = random_tuner.tune(X, y)
    print(f"Best parameters: {random_result.best_params}")
    print(f"Best score: {random_result.best_score:.4f}")

    # Test Bayesian Optimization
    print("\n3. Testing Bayesian Optimization...")
    bayesian_tuner = create_tuner(
        method="bayesian",
        model_class=RandomForestClassifier,
        param_space=param_distributions,
        n_trials=15,
        scoring="f1",
        cv=3,
    )

    bayesian_result = bayesian_tuner.tune(X, y)
    print(f"Best parameters: {bayesian_result.best_params}")
    print(f"Best score: {bayesian_result.best_score:.4f}")

    # Save results
    bayesian_result.save("./test_output/tuning_results.json")
    print("\nTuning results saved!")


def test_experiment_runner():
    """Test the experiment runner with multiple models"""
    print("\n" + "=" * 60)
    print("Testing Experiment Runner")
    print("=" * 60)

    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, n_classes=2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create experiment suite
    parser = ConfigParser()
    experiments = []

    # Add different models to compare
    for model_type in [ModelType.RANDOM_FOREST, ModelType.FNN, ModelType.LSTM]:
        config = parser.create_default_config()
        config.experiment_name = f"test_{model_type.value}"
        config.model_type = model_type
        config.output_dir = "./test_output/experiments"

        # Optimize for hardware
        config = optimize_config_for_hardware(config)

        # Adjust for smaller dataset
        config.training.epochs = 10
        config.training.batch_size = 32

        experiments.append((config.experiment_name, config))

    # Create suite
    suite = ExperimentSuite(
        name="test_model_comparison",
        experiments=experiments,
        output_dir="./test_output/experiments",
        parallel=False,  # Sequential for testing
        tune_hyperparameters=False,  # Disable for faster testing
        tuning_trials=5,
    )

    # Run experiments
    runner = ExperimentRunner()
    results = runner.run_experiment_suite(suite, X_train, y_train, X_test, y_test)

    # Display results
    print("\nExperiment Results:")
    print("-" * 40)
    for result in results:
        if result.error_message:
            print(f"{result.config_name}: FAILED - {result.error_message}")
        else:
            print(f"{result.config_name}:")
            print(f"  F1-Score: {result.metrics.get('f1_score', 0):.4f}")
            print(f"  Accuracy: {result.metrics.get('accuracy', 0):.4f}")
            print(f"  Training time: {result.training_time:.2f}s")

    # Compare experiments
    comparison_df = runner.compare_experiments(results)
    print("\nModel Comparison:")
    print(comparison_df.to_string())

    # Create ensemble if we have enough successful models
    successful_results = [r for r in results if r.error_message is None]
    if len(successful_results) >= 2:
        print("\nCreating ensemble from results...")
        ensemble_config = parser.create_default_config()
        ensemble_config.experiment_name = "test_ensemble"
        ensemble_config.output_dir = "./test_output/experiments"

        ensemble_result = runner.create_ensemble_from_results(
            successful_results[:2],  # Use first 2 models
            ensemble_config,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        print(f"Ensemble F1-Score: {ensemble_result.metrics.get('f1_score', 0):.4f}")


def test_distributed_training():
    """Test distributed training capabilities"""
    print("\n" + "=" * 60)
    print("Testing Distributed Training")
    print("=" * 60)

    # This test requires specific hardware setup
    # For now, we'll just test the initialization

    from training import DistributedTrainer

    # Create configuration
    parser = ConfigParser()
    config = parser.create_default_config()
    config.experiment_name = "test_distributed"
    config.model_type = ModelType.CNN

    # Create distributed trainer
    try:
        distributed_trainer = DistributedTrainer(config)
        print("Distributed trainer created successfully!")
        print(f"Strategy: {distributed_trainer.strategy}")
    except Exception as e:
        print(f"Distributed training not available: {e}")


def test_model_checkpointing():
    """Test model checkpointing and recovery"""
    print("\n" + "=" * 60)
    print("Testing Model Checkpointing")
    print("=" * 60)

    from training import ModelCheckpointer

    # Create checkpointer
    checkpointer = ModelCheckpointer(
        checkpoint_dir="./test_output/checkpoints", model_name="test_model"
    )

    # Test sklearn model checkpointing
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=10, random_state=42)

    # Train on dummy data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)

    # Save checkpoint
    metrics = {"accuracy": 0.95, "f1_score": 0.94}
    checkpoint_path = checkpointer.save_sklearn_model(model, metrics)
    print(f"Model saved to: {checkpoint_path}")

    # Load checkpoint
    loaded_model = checkpointer.load_best_model("sklearn")
    print("Model loaded successfully!")

    # Verify it works
    predictions = loaded_model.predict(X[:10])
    print(f"Test predictions: {predictions}")


def create_experiment_suite_yaml():
    """Create an example experiment suite YAML file"""
    suite_config = {
        "name": "weather_prediction_suite",
        "output_dir": "./test_output/suite_results",
        "parallel": True,
        "max_workers": 2,
        "tune_hyperparameters": True,
        "tuning_method": "random",
        "tuning_trials": 10,
        "experiments": [
            {
                "name": "rf_baseline",
                "config": {
                    "experiment_name": "rf_baseline",
                    "model_type": "random_forest",
                    "random_forest": {"n_estimators": 100, "max_depth": 10},
                    "training": {"batch_size": 32, "epochs": 50},
                },
            },
            {
                "name": "lstm_advanced",
                "config": {
                    "experiment_name": "lstm_advanced",
                    "model_type": "lstm",
                    "lstm": {"units": [64, 32], "dropout": 0.2, "bidirectional": True},
                    "training": {"batch_size": 64, "epochs": 100, "early_stopping": True},
                },
            },
            {
                "name": "ensemble_best",
                "config": {
                    "experiment_name": "ensemble_best",
                    "model_type": "ensemble",
                    "ensemble": {
                        "method": "voting",
                        "voting": "soft",
                        "base_models": ["random_forest", "fnn"],
                    },
                },
            },
        ],
    }

    import yaml

    with open("./test_output/experiment_suite.yaml", "w") as f:
        yaml.dump(suite_config, f, default_flow_style=False)

    print("Example experiment suite YAML created at: ./test_output/experiment_suite.yaml")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE TRAINING PIPELINE")
    print("=" * 60)

    # Create output directory
    import os

    os.makedirs("./test_output", exist_ok=True)

    # Run tests
    tests = [
        ("Unified Trainer", test_unified_trainer),
        ("Hyperparameter Tuning", test_hyperparameter_tuning),
        ("Experiment Runner", test_experiment_runner),
        ("Distributed Training", test_distributed_training),
        ("Model Checkpointing", test_model_checkpointing),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"\n✓ {test_name} - PASSED")
        except Exception as e:
            print(f"\n✗ {test_name} - FAILED: {str(e)}")
            import traceback

            traceback.print_exc()

    # Create example YAML
    create_experiment_suite_yaml()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
    print("\nCheck the ./test_output directory for results and reports.")


if __name__ == "__main__":
    run_all_tests()
