"""
Integrated Weather Regulation Prediction Model

This module provides the main entry point for training and evaluating
weather-based regulation prediction models. It integrates:
- Data loading and validation
- Feature engineering
- Preprocessing pipelines
- Multiple model architectures
- Experiment tracking and management
"""

import datetime as dt
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Legacy imports for backward compatibility
from functions import (
    FNN_model,
    graphs_weather,
    labelbinarizer_CC_preprocess,
    labelbinarizer_WD_preprocess,
    minmaxscaler_preprocess,
)

# New modular imports
from config import DataConfig, ExperimentConfig
from config_utils import load_config

# Data pipeline imports
from data.data_loader import DataLoader
from data.data_validation import DataValidator
from data.feature_engineering import AutomatedFeatureEngineer
from data.preprocessing import PreprocessingPipeline
from run_experiments import ExperimentRunner

# Model imports
# Training imports

# Configure warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Dly_Classifier:
    """
    Weather-based ATFM Regulation Classifier

    This class computes the probability of regulation existence due to weather
    events at airports within the NMOC region. It supports both legacy and
    new modular approaches.

    Inputs:
        1. METAR: Meteorological reports (30-min intervals)
        2. TAF: Terminal Aerodrome Forecasts (enhanced predictions)
        3. Regulations: ATFM regulations dataset

    Output:
        - Regulation probability for each airport
        - Model performance metrics
        - Feature importance analysis

    Supported Models:
        - Random Forest
        - LSTM (multiple variants)
        - CNN, RNN, FNN
        - Transformer, GRU
        - Ensemble methods
        - And more...
    """

    def __init__(
        self,
        time_init,
        time_end,
        info_path,
        Output_path,
        AD,
        time_delta,
        download_type,
        config: str | ExperimentConfig | None = None,
        use_new_pipeline: bool = True,
    ):
        """
        Initialize the classifier

        Args:
            time_init: Start datetime
            time_end: End datetime
            info_path: Path to input data
            Output_path: Path for output files
            AD: Airport ICAO code
            time_delta: Time step in minutes
            download_type: 0=download from web, 1=use local files
            config: Configuration file path or ExperimentConfig object
            use_new_pipeline: Whether to use new modular pipeline
        """
        self.time_init = time_init
        self.time_end = time_end
        self.info_path = Path(info_path)
        self.Output_path = Path(Output_path)
        self.AD = AD
        self.time_delta = time_delta
        self.download_type = download_type
        self.use_new_pipeline = use_new_pipeline

        # Setup logging
        self.logger = self._setup_logger()

        # Load or create configuration
        if config:
            if isinstance(config, str):
                self.config = load_config(config)
            else:
                self.config = config
        else:
            # Create default configuration
            self.config = self._create_default_config()

        # Initialize components if using new pipeline
        if self.use_new_pipeline:
            self._initialize_pipeline_components()

        # Load data based on pipeline choice
        if self.use_new_pipeline:
            self.logger.info("Using new modular pipeline")
            self._load_data_new_pipeline()
        else:
            self.logger.info("Using legacy pipeline")
            self._load_data_legacy_pipeline()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _create_default_config(self) -> ExperimentConfig:
        """Create default configuration based on initialization parameters"""
        return ExperimentConfig(
            name=f"experiment_{self.AD}_{self.time_init.strftime('%Y%m%d')}",
            description=f"Weather regulation prediction for {self.AD}",
            data=DataConfig(
                airports=[self.AD],
                start_date=self.time_init.strftime("%Y-%m-%d"),
                end_date=self.time_end.strftime("%Y-%m-%d"),
                time_step_minutes=self.time_delta,
                data_path=str(self.info_path),
                output_path=str(self.Output_path),
            ),
        )

    def _initialize_pipeline_components(self):
        """Initialize new pipeline components"""
        self.data_loader = DataLoader(self.config, cache_enabled=True)
        self.data_validator = DataValidator(self.config.data)
        self.feature_engineer = AutomatedFeatureEngineer(self.config.data)
        self.preprocessor = PreprocessingPipeline(self.config)
        self.experiment_runner = ExperimentRunner(self.config)

    def _load_data_new_pipeline(self):
        """Load data using new modular pipeline"""
        self.logger.info("Loading data with new pipeline...")

        # Load all data types
        self.data = self.data_loader.load_all_data(
            airports=[self.AD], start_date=self.time_init, end_date=self.time_end
        )

        # Validate data
        self.logger.info("Validating data...")
        validation_results = self.data_validator.validate_dataset(
            self.data["features"], check_anomalies=True
        )

        # Log validation results
        for name, result in validation_results.items():
            if not result.is_valid:
                self.logger.warning(f"{name} validation failed: {result.errors}")
            if result.warnings:
                self.logger.info(f"{name} warnings: {result.warnings}")

        # Store processed data
        self.features = self.data["features"]
        self.X = None  # Will be set after preprocessing
        self.y = (
            self.features["has_regulation"].values
            if "has_regulation" in self.features.columns
            else None
        )

    def _load_data_legacy_pipeline(self):
        """Load data using legacy pipeline for backward compatibility"""
        # Legacy loading code (preserved from original)
        x = pd.read_csv(
            os.path.join(self.info_path, "METAR", f"METAR_{self.AD}_filtered_final.csv")
        )
        self.X = x.iloc[:, 1:15].values

        # Compute target vector
        y = pd.read_csv(
            os.path.join(self.info_path, "Regulations", "Regulation_binarizer_2_30.csv")
        )
        self.y = y[f"{self.AD}"].values

    def train_models(
        self, models: list[str] | None = None, use_hyperparameter_tuning: bool = True
    ) -> dict[str, Any]:
        """
        Train specified models using the configured pipeline

        Args:
            models: List of model names to train (None = all available)
            use_hyperparameter_tuning: Whether to perform hyperparameter tuning

        Returns:
            Dictionary of results for each model
        """
        if self.use_new_pipeline:
            return self._train_models_new_pipeline(models, use_hyperparameter_tuning)
        else:
            return self._train_models_legacy_pipeline()

    def _train_models_new_pipeline(
        self, models: list[str] | None = None, use_hyperparameter_tuning: bool = True
    ) -> dict[str, Any]:
        """Train models using new modular pipeline"""
        self.logger.info("Training models with new pipeline")

        # Preprocess features if not already done
        if self.X is None and self.features is not None:
            self.logger.info("Preprocessing features...")

            # Detect feature types
            numerical_features = self.features.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = self.features.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Remove target column if present
            if "has_regulation" in numerical_features:
                numerical_features.remove("has_regulation")
            if "has_regulation" in categorical_features:
                categorical_features.remove("has_regulation")

            # Create preprocessing pipeline
            self.pipeline = self.preprocessor.create_pipeline(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                feature_selection=True,
                task="classification",
            )

            # Fit and transform
            X_df = self.features.drop(
                columns=["has_regulation"] if "has_regulation" in self.features.columns else []
            )
            self.X = self.preprocessor.fit_transform(X_df, self.y)

        # Configure models to train
        if models is None:
            models = ["random_forest", "lstm", "transformer", "ensemble"]

        # Update experiment config with selected models
        self.config.models = models

        # Run experiments
        self.logger.info(f"Training models: {models}")
        results = self.experiment_runner.run_experiment(
            data=(self.X, self.y), hyperparameter_tuning=use_hyperparameter_tuning
        )

        # Generate report
        report_path = (
            self.Output_path
            / f"experiment_report_{self.AD}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        self.experiment_runner.generate_report(str(report_path))
        self.logger.info(f"Report saved to: {report_path}")

        return results

    def _train_models_legacy_pipeline(self) -> dict[str, Any]:
        """Train models using legacy pipeline for backward compatibility"""
        self.logger.info("Training models with legacy pipeline")
        results = {}

        # Example: Train FNN model (currently active in original code)
        self.logger.info("Training FNN model...")
        precision, recall, accuracy, f1_score = FNN_model(
            self.X,
            self.y,
            hidden_layer_sizes=(30, 30, 30),
            activation="tanh",
            solver="adam",
            max_iter=2000,
            test_size=0.20,
            random_state=42,
        )

        results["FNN"] = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1_score": f1_score,
        }

        # Save results
        df_results = pd.DataFrame([results["FNN"]])
        df_results.to_csv(self.Output_path / f"FNN_{self.AD}_results.csv", index=False)

        return results

        # image_paths = []
        # for i in range(len(n_trees)):
        #     for j in range(len(criterion)):
        #         image_paths.append(
        #             f'.\Output\RF\RF_{self.AD}_CM_{n_trees[i]}trees_{criterion[j]}C_7MD.png')
        # # Abre las imágenes y las almacena en una lista
        # images = [Image.open(path) for path in image_paths]

        # # Asegúrate de que todas las imágenes tengan el mismo tamaño
        # width, height = images[0].size
        # for img in images:
        #     if img.size != (width, height):
        #         raise ValueError("Todas las imágenes deben tener el mismo tamaño.")

        # # Crea una imagen en blanco con el tamaño de la matriz 3x3
        # combined_image = Image.new("RGB", (3 * width, 2 * height))

        # # Combina las imágenes en la matriz 3x3
        # for i in range(2):
        #     for j in range(3):
        #         index = i * 3 + j
        #         combined_image.paste(images[index], (j * width, i * height))

        # # Guarda la imagen combinada
        # combined_image.save(
        #     f".\Output\RF\RF_CM_{self.AD}_ginientropylog_{n_trees[0]}{n_trees[1]}.png")

        # # Cierra las imágenes originales
        # for img in images:
        #     img.close()

        # # Crear un DataFrame con los resultados
        # df_results = pd.DataFrame(results, columns=[
        #                         'n_trees', 'criterion', 'precision', 'recall', 'accuracy', 'f1_score'])

        # # Guardar el DataFrame en un archivo CSV
        # df_results.to_csv(
        #     f'.\Output\RF\RF_{self.AD}_results_7MD_2050.csv', index=False)

        # epochs = [30]
        # batch_size = [40]
        # print('RNN:')
        # results = []
        # for i in range(len(batch_size)):
        #     for j in range(len(epochs)):
        #         precision, recall, accuracy, f1_score = RNN_model(X, Y, epochs=epochs[j], batch_size=batch_size[i],
        #                                                         AD=self.AD, test_size=0.20, random_state=42,
        #                                                         optimizer='adam', activation='sigmoid',
        #                                                         loss='binary_crossentropy',
        #                                                         Output_path=self.Output_path)
        #          # Agregar los resultados y los valores de n_trees y criterion a las listas
        #         results.append([batch_size[i], epochs[j], precision, recall, accuracy, f1_score])

        # # Crear un DataFrame con los resultados
        # df_results = pd.DataFrame(results, columns=[
        #                         'batch', 'epochs', 'precision', 'recall', 'accuracy', 'f1_score'])

        # # Guardar el DataFrame en un archivo CSV
        # df_results.to_csv(f'.\Output\RNN\RNN_{self.AD}_results_', index=False)
        # print('LSTM:')
        # results = []
        # units = [100,50,100]
        # epochs = [20,50,100]
        # for i in range(len(units)):
        #      for j in range(len(epochs)):
        #         LSTM_model_1(X, Y, epochs=epochs[j],
        #                             batch_size = 40,
        #                             AD = self.AD,
        #                             test_size = 0.20,
        #                             random_state = 42,
        #                             optimizer = 'adam',
        #                             activation = 'sigmoid',
        #                             loss = 'binary_crossentropy',
        #                             dropout_rate = 0.3,
        #                             units = units[i],
        #                             Output_path = os.path.join(self.Output_path,'LSTM'))
        # results.append([units[i], epochs[j],
        #                  precision, recall, accuracy, f1_score])

        # image_paths = []
        # for i in range(len(units)):
        #     for j in range(len(epochs)):
        #         image_paths.append(
        #             f'.\Output\LSTM\LSTM_{self.AD}_loss_{epochs[j]}E_32B_{units[i]}U.png')
        # # Abre las imágenes y las almacena en una lista
        # images = [Image.open(path) for path in image_paths]

        # # Asegúrate de que todas las imágenes tengan el mismo tamaño
        # width, height = images[0].size
        # for img in images:
        #     if img.size != (width, height):
        #         raise ValueError("Todas las imágenes deben tener el mismo tamaño.")

        # # Crea una imagen en blanco con el tamaño de la matriz 3x3
        # combined_image = Image.new("RGB", (3 * width, 3 * height))

        # # Combina las imágenes en la matriz 3x3
        # for i in range(3):
        #     for j in range(3):
        #         index = i * 3 + j
        #         combined_image.paste(images[index], (j * width, i * height))

        # # Guarda la imagen combinada
        # combined_image.save(f".\Output\LSTM\LSTM_{self.AD}_loss_combined.png")

        # # Cierra las imágenes originales
        # for img in images:
        #     img.close()

        # # Crear un DataFrame con los resultados
        # df_results = pd.DataFrame(results, columns=['units', 'epochs', 'precision', 'recall', 'accuracy', 'f1_score'])

        # # Guardar el DataFrame en un archivo CSV
        # df_results.to_csv(f'.\Output\LSTM\LSTM_{self.AD}_results_TEST.csv', index=False)

        # print('WaveNet:')
        # WaveNet_model(X, Y, epochs=60, batch_size=32, AD=self.AD, test_size=0.25,
        #               random_state=50, optimizer='adam', activation='sigmoid', loss='binary_crossentropy', Output_path=self.Output_path)

    def run_legacy_experiment(self):
        """Run experiment using legacy approach (for backward compatibility)"""
        self.logger.info("Running legacy experiment...")

        # Load data
        self._load_data_legacy_pipeline()

        # Train models
        results = self._train_models_legacy_pipeline()

        # Print results
        for model_name, metrics in results.items():
            self.logger.info(f"\n{model_name} Results:")
            for metric, value in metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")

        return results

    def run_modular_experiment(
        self, models: list[str] | None = None, hyperparameter_tuning: bool = True
    ):
        """Run experiment using new modular approach"""
        self.logger.info("Running modular experiment...")

        # Load and validate data
        self._load_data_new_pipeline()

        # Engineer features
        if self.config.data.feature_engineering:
            self.logger.info("Engineering features...")
            self.features = self.feature_engineer.fit_transform(self.features, self.y)

        # Train models
        results = self.train_models(models=models, use_hyperparameter_tuning=hyperparameter_tuning)

        return results

        # print('CNN-RNN:')
        # cm_CNN_RNN = CNN_RNN_model(X, Y, epochs=60, batch_size=32,
        #                            AD=self.AD, time_init=self.time_init, time_end=self.time_end,
        #                            test_size=0.25, random_state=50, activationDense='sigmoid', activationConv1D='relu',
        #                             optimizer='adam', loss='binary_crossentropy', kernel_size=3, pool_size=2,
        #                             Conv1D_layer=64, Dense_layer=1, Output_path=self.Output_path)

        # print('CNN:')
        # cm_CNN = CNN_model(X, Y, epochs=50, batch_size=36,AD=self.AD,time_init=self.time_init,
        #    time_end=self.time_end, test_size=0.2, random_state=42,
        #    activationDense='sigmoid', activationConv1D='relu',
        #    optimizer='adam', loss='binary_crossentropy', kernel_size=3,
        #    pool_size=2, Conv1D_layer=64, Dense_layer=1, Output_path=self.Output_path)

        # print('CNN-LSTM:')
        # cm_CNN_LSTM = CNN_LSTM_model(X, Y, epochs=60, batch_size=32,
        #                            AD=self.AD, time_init=self.time_init, time_end=self.time_end,
        #                            test_size=0.25, random_state=50, activationDense='sigmoid', activationConv1D='relu',
        #                            optimizer='adam', loss='binary_crossentropy', kernel_size=3, pool_size=2,
        #                            Conv1D_layer=64, Dense_layer=1, Output_path=self.Output_path)

        # # Save the model results
        # confusion_matrices = {
        #     'Random Forest': cm_RF,
        #     'Recurrent Neural Network': cm_RNN,
        #     'LSTM': cm_LSTM,
        #     'FNN': cm_FNN,
        #     # 'CNN-RNN': cm_CNN_RNN,
        #     'CNN': cm_CNN,
        #     # 'CNN-LSTM': cm_CNN_LSTM
        # }

        # Nombre del archivo de texto
        # results = os.path.join(f'.\Output\CM_{self.AD}_{time_init.strftime("%Y%m%d")}_{time_end.strftime("%Y%m%d")}.txt')

        # # Abrir el archivo en modo de escritura
        # with open(results, 'w') as file:
        #     # Iterar a través del diccionario y escribir el nombre y la matriz de confusión
        #     for model_name, confusion_matrix in confusion_matrices.items():
        #         file.write(f'{model_name}:\n')
        #         file.write(f'{confusion_matrix}\n\n')

        # Plot the confusion matrices

    def filtered_df(self, option, time_delta=False, download_type=False):
        """
        This function creates a filtered dataframe.
        The filters are selected according to the specifications.
        In particular, for this project these are:

            1. Aerodrome
            2. METAR
        """
        if option == 1:
            reg_path = os.path.join(self.info_path, "Regulations", "List of Regulations.csv")

            # Plot graphs and filter regulations by aerodrome, weather and assign 6 labels to different weather types
            self.df_reg = graphs_weather(reg_path, self.AD)
            return self.df_reg

        elif option == 2:
            # if download_type == 0:
            #     airports_NMOC_fun(self.df_reg_4)
            #     download_METARs(os.path.join(self.info_path, 'Regulations', 'Airports_regulation_list.csv'),
            #                     self.time_init, self.time_end,
            #                     output_csv_file=os.path.join(self.info_path,'METAR','Airports_METAR.csv'))
            #     self.df_MET = filter_METARs(data_path=os.path.join(self.info_path, 'METAR', 'Airports_METAR.csv'),
            #                 data_path_out=os.path.join(self.info_path, 'METAR', 'Airports_METAR_filtered.csv'),
            #                                 time_step=time_delta, download_type=download_type)
            #     #self.df_MET = pd.read_csv(MET_path,sep=',')
            #     return self.df_MET
            if download_type == 1:
                # self.df_MET = filter_METARs(data_path=os.path.join(
                #     self.info_path, 'METAR', f'METAR_{self.AD}_{self.time_init.strftime("%Y_%m_%d")}_{self.time_end.strftime("%Y_%m_%d")}.csv'),
                #                             data_path_out=os.path.join(
                #                                 self.info_path, 'METAR', f'METAR_{self.AD}_filtered.csv'),
                #                             time_step=time_delta, download_type=download_type)
                # self.df_MET = filter_by_dates(os.path.join(self.info_path, 'Regulations', f'Regulation_binarizer_2_30.csv'),
                #                               os.path.join(self.info_path, 'METAR', f'METAR_{self.AD}_filtered.csv'))
                # self.df_MET.to_csv(os.path.join( '.\Data\METAR',f'METAR_{self.AD}_filtered_dates.csv'), index_label=False)

                _, _, self.df_MET = minmaxscaler_preprocess(
                    os.path.join(self.info_path, "METAR", f"METAR_{self.AD}_filtered_dates.csv"),
                    [
                        "Horizontal visibility",
                        "Ceiling height",
                        "Wind Speed",
                        "Temperature",
                        "Dewpoint",
                    ],
                    os.path.join(self.info_path, "METAR", f"METAR_{self.AD}_filtered_minmax.csv"),
                )

                # We need to add Ceiling coverage and wind direction category to the filtered METAR
                # Perform pre-processing method for wind direction
                self.df_MET_WD = labelbinarizer_WD_preprocess(
                    os.path.join(self.info_path, "METAR", f"METAR_{self.AD}_filtered_dates.csv"),
                    "Wind Direction",
                    "WD",
                    os.path.join(r".\Data\METAR", f"METAR_{self.AD}_WDCat.csv"),
                )

                # Perform pre-processing method for ceiling coverage
                self.df_MET_CC = labelbinarizer_CC_preprocess(
                    os.path.join(self.info_path, "METAR", f"METAR_{self.AD}_filtered_dates.csv"),
                    "CCCat",
                    os.path.join(r".\Data\METAR", f"METAR_{self.AD}_CCCat.csv"),
                )

                # Obtain final dataframe
                file1 = pd.read_csv(os.path.join(r".\Data\METAR", f"METAR_{self.AD}_CCCat.csv"))
                file2 = pd.read_csv(os.path.join(r".\Data\METAR", f"METAR_{self.AD}_WDCat.csv"))
                file3 = pd.read_csv(
                    os.path.join(self.info_path, "METAR", f"METAR_{self.AD}_filtered_minmax.csv")
                )
                self.df_MET_final = pd.concat([file3, file1, file2], axis=1, ignore_index=True)
                self.df_MET_final.to_csv(
                    os.path.join(r".\Data\METAR", f"METAR_{self.AD}_filtered_final.csv"),
                    index=False,
                )

                return self.df_MET_final
            else:
                print(
                    f"Error (download type={download_type}). Select: 0=download from web, 1=download txt"
                )

        else:
            print(f"Error (option={option}). Select an option: 1=Regulations, 2=METAR")

        # return compute_Dly_Classifier(reg_processed_path,)


####################################################################
####################################################################
####################  Model computation   ##########################
####################################################################
####################################################################

time_init = dt.datetime(2017, 1, 1, 0, 50)
time_end = dt.datetime(2019, 12, 8, 23, 50)
Data_path = r".\Data"
Output_path = r".\Output"
aerodrome = "EGLL"
download_type = 1  # Download type for METAR
time_sep = 30  # Measurements time step (multiple of 5 minutes)
# Call the class by moving one folder back
if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Weather-based ATFM Regulation Prediction")
    parser.add_argument(
        "--airport", "-a", type=str, default="EGLL", help="Airport ICAO code (default: EGLL)"
    )
    parser.add_argument("--config", "-c", type=str, help="Configuration file path")
    parser.add_argument(
        "--pipeline",
        choices=["legacy", "modular", "compare"],
        default="legacy",
        help="Which pipeline to use (default: legacy for backward compatibility)",
    )
    parser.add_argument(
        "--models", "-m", nargs="+", help="Models to train (e.g., lstm transformer ensemble)"
    )
    parser.add_argument("--no-tuning", action="store_true", help="Disable hyperparameter tuning")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    # Use command line args or defaults
    if args.start_date:
        time_init = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        time_end = dt.datetime.strptime(args.end_date, "%Y-%m-%d").replace(hour=23, minute=59)

    if args.airport:
        aerodrome = args.airport

    # Create and run classifier
    if args.pipeline == "legacy" and not args.config:
        # Run legacy mode for backward compatibility
        target = Dly_Classifier(
            time_init,
            time_end,
            Data_path,
            Output_path,
            aerodrome,
            time_sep,
            download_type,
            use_new_pipeline=False,
        )
        print(target)
    else:
        # Run with new features
        classifier = Dly_Classifier(
            time_init=time_init,
            time_end=time_end,
            info_path=Data_path,
            Output_path=Output_path,
            AD=aerodrome,
            time_delta=time_sep,
            download_type=download_type,
            config=args.config,
            use_new_pipeline=(args.pipeline == "modular"),
        )

        # Run experiment
        if args.pipeline == "modular":
            results = classifier.run_modular_experiment(
                models=args.models, hyperparameter_tuning=not args.no_tuning
            )
        else:
            results = classifier.run_legacy_experiment()

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED")
        print("=" * 60)
