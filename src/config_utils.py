"""
Configuration utilities for the weather regulation prediction system.
Provides helper functions and CLI tools for configuration management.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

from tabulate import tabulate

from .config import ExperimentConfig, validate_config
from .config_parser import ConfigParser, load_config, save_config


class ConfigurationManager:
    """High-level configuration management utilities."""

    def __init__(self):
        self.parser = ConfigParser()
        self.configs_dir = Path("configs")
        self.ensure_configs_dir()

    def ensure_configs_dir(self):
        """Ensure configs directory exists."""
        self.configs_dir.mkdir(exist_ok=True)

    def list_configs(self) -> list[dict[str, Any]]:
        """List all available configuration files."""
        configs = []

        for config_file in self.configs_dir.glob("*.yaml"):
            try:
                config = load_config(config_file)
                configs.append(
                    {
                        "file": config_file.name,
                        "name": config.name,
                        "description": (
                            config.description[:50] + "..."
                            if len(config.description) > 50
                            else config.description
                        ),
                        "models": ", ".join(config.models_to_train),
                        "airports": ", ".join(config.data.airports),
                    }
                )
            except Exception as e:
                configs.append(
                    {
                        "file": config_file.name,
                        "name": "ERROR",
                        "description": str(e),
                        "models": "-",
                        "airports": "-",
                    }
                )

        return configs

    def create_config_wizard(self) -> ExperimentConfig:
        """Interactive wizard to create new configuration."""
        print("\n=== Configuration Creation Wizard ===\n")

        # Basic information
        name = input("Configuration name: ").strip()
        description = input("Description: ").strip()

        # Data settings
        print("\n--- Data Settings ---")
        airports_str = input("Airports (comma-separated, e.g., EGLL,LSZH): ").strip()
        airports = [a.strip() for a in airports_str.split(",")]

        time_delta = int(input("Time delta in minutes (default 30): ") or "30")
        test_size = float(input("Test size (0.0-1.0, default 0.2): ") or "0.2")

        # Model selection
        print("\n--- Model Selection ---")
        print("Available models:")
        models = ["random_forest", "lstm", "cnn", "gru", "transformer", "ensemble", "autoencoder"]
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")

        selected_indices = input("Select models (comma-separated numbers): ").strip()
        selected_models = []
        for idx in selected_indices.split(","):
            try:
                selected_models.append(models[int(idx.strip()) - 1])
            except (ValueError, IndexError):
                pass

        # Hyperparameter tuning
        print("\n--- Hyperparameter Tuning ---")
        use_tuning = input("Enable hyperparameter tuning? (y/n): ").strip().lower() == "y"
        tuning_method = "grid"
        tuning_trials = 100

        if use_tuning:
            print("Tuning methods: 1. grid, 2. random, 3. bayesian")
            method_idx = input("Select method (1-3): ").strip()
            tuning_method = (
                ["grid", "random", "bayesian"][int(method_idx) - 1]
                if method_idx.isdigit()
                else "grid"
            )
            tuning_trials = int(input("Number of trials (default 100): ") or "100")

        # Create configuration
        config = ExperimentConfig(name=name, description=description)

        config.data.airports = airports
        config.data.time_delta = time_delta
        config.data.test_size = test_size

        config.models_to_train = selected_models
        config.hyperparameter_tuning = use_tuning
        config.tuning_method = tuning_method
        config.tuning_trials = tuning_trials

        return config

    def optimize_config_for_hardware(
        self, config: ExperimentConfig, gpu_memory_gb: int = 8
    ) -> ExperimentConfig:
        """
        Optimize configuration based on available hardware.

        Args:
            config: Base configuration
            gpu_memory_gb: Available GPU memory in GB

        Returns:
            Optimized configuration
        """
        optimized = copy.deepcopy(config)

        # Adjust batch sizes based on GPU memory
        memory_factor = gpu_memory_gb / 8.0  # Normalize to 8GB baseline

        for model_name in ["lstm", "cnn", "gru", "transformer"]:
            model_config = getattr(optimized, model_name, None)
            if model_config and hasattr(model_config, "batch_size"):
                # Scale batch sizes
                original_batch_sizes = model_config.batch_size
                model_config.batch_size = [int(bs * memory_factor) for bs in original_batch_sizes]

        # Enable mixed precision for newer GPUs
        if gpu_memory_gb >= 8:
            optimized.training.mixed_precision = True

        # Adjust number of workers
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        optimized.training.num_workers = min(cpu_count - 1, 8)

        return optimized

    def generate_experiment_report(self, config: ExperimentConfig) -> str:
        """Generate a detailed experiment report."""
        report = []
        report.append("# Experiment Configuration Report")
        report.append(f"\n## {config.name}")
        report.append(f"**Description:** {config.description}")
        report.append(f"**Version:** {config.version}")

        # Data configuration
        report.append("\n### Data Configuration")
        report.append(f"- **Airports:** {', '.join(config.data.airports)}")
        report.append(f"- **Time Period:** {config.data.time_init} to {config.data.time_end}")
        report.append(f"- **Time Delta:** {config.data.time_delta} minutes")
        report.append(
            f"- **Data Split:** Train {1-config.data.test_size-config.data.validation_size:.0%} / Val {config.data.validation_size:.0%} / Test {config.data.test_size:.0%}"
        )

        # Models
        report.append("\n### Models to Train")
        for model in config.models_to_train:
            report.append(f"- {model}")
            model_config = getattr(config, model, None)
            if model_config:
                # Show key hyperparameters
                if hasattr(model_config, "epochs"):
                    report.append(f"  - Epochs: {model_config.epochs}")
                if hasattr(model_config, "batch_size"):
                    report.append(f"  - Batch size: {model_config.batch_size}")

        # Training settings
        report.append("\n### Training Configuration")
        report.append(
            f"- **Cross-validation:** {'Yes' if config.training.use_cross_validation else 'No'} ({config.training.cv_folds} folds)"
        )
        report.append(
            f"- **Early stopping:** {'Yes' if config.training.use_early_stopping else 'No'} (patience: {config.training.early_stopping_patience})"
        )
        report.append(
            f"- **Hardware:** GPU {'enabled' if config.training.use_gpu else 'disabled'}, Mixed precision {'on' if config.training.mixed_precision else 'off'}"
        )

        # Hyperparameter tuning
        if config.hyperparameter_tuning:
            report.append("\n### Hyperparameter Tuning")
            report.append(f"- **Method:** {config.tuning_method}")
            report.append(f"- **Trials:** {config.tuning_trials}")

            # Calculate total combinations
            for model in config.models_to_train:
                grid = self.parser.generate_hyperparameter_grid(config, model)
                report.append(f"- **{model} combinations:** {len(grid)}")

        return "\n".join(report)

    def estimate_training_time(self, config: ExperimentConfig) -> dict[str, float]:
        """
        Estimate training time for each model.

        Returns:
            Dictionary with model names and estimated hours
        """
        estimates = {}

        # Base time estimates per epoch per 10k samples (in minutes)
        base_times = {
            "random_forest": 0.1,  # Very fast, not epoch-based
            "lstm": 2.0,
            "cnn": 1.5,
            "gru": 1.8,
            "transformer": 3.0,
            "ensemble": 0.5,  # Depends on base models
            "autoencoder": 1.2,
        }

        # Estimate dataset size (rough approximation)
        days = (config.data.time_end - config.data.time_init).days
        samples_per_day = 24 * 60 / config.data.time_delta
        total_samples = days * samples_per_day * len(config.data.airports)
        sample_factor = total_samples / 10000

        for model in config.models_to_train:
            base_time = base_times.get(model, 1.0)
            model_config = getattr(config, model, None)

            if model == "random_forest":
                # Tree-based models scale differently
                n_estimators = max(getattr(model_config, "n_estimators", [100]))
                estimates[model] = base_time * n_estimators / 100 * sample_factor / 60
            else:
                # Neural networks
                epochs = max(getattr(model_config, "epochs", [100]))
                estimates[model] = base_time * epochs * sample_factor / 60

            # Add overhead for hyperparameter tuning
            if config.hyperparameter_tuning:
                if config.tuning_method == "grid":
                    grid_size = len(self.parser.generate_hyperparameter_grid(config, model))
                    estimates[model] *= grid_size
                elif config.tuning_method == "random":
                    estimates[model] *= config.tuning_trials / 10
                elif config.tuning_method == "bayesian":
                    estimates[model] *= config.tuning_trials / 20

        return estimates


def create_cli():
    """Create command-line interface for configuration management."""
    parser = argparse.ArgumentParser(
        description="Configuration management for weather regulation prediction"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List configs command
    subparsers.add_parser("list", help="List available configurations")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument("config", help="Configuration file path")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create new configuration")
    create_parser.add_argument("--wizard", action="store_true", help="Use interactive wizard")
    create_parser.add_argument("--output", "-o", help="Output file path")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two configurations")
    compare_parser.add_argument("config1", help="First configuration file")
    compare_parser.add_argument("config2", help="Second configuration file")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate configuration report")
    report_parser.add_argument("config", help="Configuration file path")
    report_parser.add_argument("--output", "-o", help="Output file path")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize for hardware")
    optimize_parser.add_argument("config", help="Configuration file path")
    optimize_parser.add_argument("--gpu-memory", type=int, default=8, help="GPU memory in GB")
    optimize_parser.add_argument("--output", "-o", help="Output file path")

    return parser


def main():
    """Main CLI entry point."""
    parser = create_cli()
    args = parser.parse_args()

    manager = ConfigurationManager()

    if args.command == "list":
        configs = manager.list_configs()
        if configs:
            print(tabulate(configs, headers="keys", tablefmt="grid"))
        else:
            print("No configuration files found")

    elif args.command == "validate":
        try:
            config = load_config(args.config)
            errors = validate_config(config)
            if errors:
                print("Validation errors:")
                for error in errors:
                    print(f"  ❌ {error}")
            else:
                print("✅ Configuration is valid!")
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")

    elif args.command == "create":
        if args.wizard:
            config = manager.create_config_wizard()
            output_path = args.output or f"configs/{config.name}.yaml"
            save_config(config, output_path)
            print(f"\n✅ Configuration saved to: {output_path}")
        else:
            print("Use --wizard flag to create configuration interactively")

    elif args.command == "compare":
        config1 = load_config(args.config1)
        config2 = load_config(args.config2)
        differences = manager.parser.compare_configs(config1, config2)

        if differences:
            print(f"\nDifferences between {args.config1} and {args.config2}:")
            for path, diff in differences.items():
                print(f"\n{path}:")
                for key, value in diff.items():
                    print(f"  {key}: {value}")
        else:
            print("Configurations are identical")

    elif args.command == "report":
        config = load_config(args.config)
        report = manager.generate_experiment_report(config)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
        else:
            print(report)

        # Add time estimates
        print("\n### Estimated Training Time")
        estimates = manager.estimate_training_time(config)
        for model, hours in estimates.items():
            print(f"- {model}: {hours:.1f} hours")

    elif args.command == "optimize":
        config = load_config(args.config)
        optimized = manager.optimize_config_for_hardware(config, args.gpu_memory)

        output_path = args.output or args.config.replace(".yaml", "_optimized.yaml")
        save_config(optimized, output_path)
        print(f"✅ Optimized configuration saved to: {output_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    import copy

    # Run CLI
    if len(sys.argv) > 1:
        main()
    else:
        # Demo usage
        manager = ConfigurationManager()

        print("Configuration Management Demo")
        print("=" * 50)

        # List configs
        print("\nAvailable configurations:")
        configs = manager.list_configs()
        print(tabulate(configs, headers="keys", tablefmt="grid"))

        # Load and report on default config
        default_config = load_config("configs/default_config.yaml")
        print("\nDefault configuration report:")
        print("-" * 50)
        report = manager.generate_experiment_report(default_config)
        print(report)

        # Time estimates
        print("\nEstimated training times:")
        estimates = manager.estimate_training_time(default_config)
        for model, hours in estimates.items():
            print(f"  {model}: {hours:.1f} hours")
